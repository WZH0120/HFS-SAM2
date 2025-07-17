import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from torchvision.transforms.functional import rgb_to_grayscale

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    kernel = kernel.to(img.device)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class SEA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(SEA, self).__init__()
        out_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, kernel_size=1),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xc = self.channel_att(x)
        xlg = xl + xg
        xla = xc * xlg
        wei = self.sig(xla)
        return wei
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class MSE(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(MSE, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1), relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), relu=False),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2), relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0), relu=False),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3), relu=False),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0), relu=False),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7, relu=False)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            BasicConv2d(in_channel, out_channel, 1, relu=False)
        )
        self.conv = BasicConv2d(5*out_channel, out_channel, 3, padding=1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        fuse = self.conv(torch.cat((x0, x1, x2, x3, x4), 1))
        x = fuse + x0
        return x


class CLA(nn.Module):
    def __init__(self, channel):
        super(CLA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = BasicConv2d(2*channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv6 = BasicConv2d(2*channel, channel, 3, padding=1)

    def forward(self, m4, m3, m2):
        m4_ = self.conv1(self.upsample(m4))
        m3_ = torch.cat([self.conv2((m3 * m4_) + m3), m4_], dim=1)
        c = self.conv6(torch.cat([self.conv5(self.conv4(m2 * self.upsample(m3)) + m2), self.conv3(self.upsample(m3_))], dim=1))
        return c


class HFS(nn.Module):
    def __init__(self, channels=64):
        super(HFS, self).__init__()
        self.sea = SEA(channels)
        self.conv1 = BasicConv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, m, f_L):
        f_L = F.interpolate(f_L, size=m.size()[2:], mode='bilinear', align_corners=True)
        m_hat = m * f_L + m
        n = self.conv1(torch.cat((m_hat, m), dim=1))
        p = self.conv2(self.sigmoid(n) * n)
        p_hat = self.sea(p)
        h = (p_hat * m) + ((-1*(torch.sigmoid(p_hat)) + 1) * m_hat)
        return h
    

class SGR(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = BasicConv2d(in_planes=in_chs, out_planes=out_chs, kernel_size=1)
        self.se = SELayer(channel=out_chs, reduction=4)
        
    def forward(self, c, h_1, h_2):
        fuse = torch.cat((c + self.down(h_1), c + h_2), dim=1)
        out = self.se(self.conv(fuse))
        return out

    
class RFM(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv1 = BasicConv2d(chs, chs, kernel_size=1)
        self.conv2 = BasicConv2d(chs, 1, kernel_size=1)

    def forward(self, m_1, s):
        x = -1*(torch.sigmoid(s)) + 1
        x = x.expand(-1, 64, -1, -1).mul(m_1)
        x = self.conv1(x)
        x = x + s
        x = self.conv2(x)
        return x


class FA_SAM2(nn.Module):
    def __init__(self) -> None:
        super(FA_SAM2, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        checkpoint_path = "./sam2_hiera_large.pt"
        model = build_sam2(model_cfg, checkpoint_path)
        # model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.mse1 = MSE(144, 64)
        self.mse2 = MSE(288, 64)
        self.mse3 = MSE(576, 64)
        self.mse4 = MSE(1152, 64)
        self.cla = CLA(64)
        self.hfs1 = HFS(64)
        self.hfs2 = HFS(64)
        self.sgr = SGR(in_chs=2*64, out_chs=64)
        self.rfm = RFM(64)
        self.out_2 = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        freq_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        f_L = freq_feature[1]
        f_1, f_2, f_3, f_4 = self.encoder(x)
        m_1, m_2, m_3, m_4 = self.mse1(f_1), self.mse2(f_2), self.mse3(f_3), self.mse4(f_4)
        c = self.cla(m_4, m_3, m_2)
        h_1 = self.hfs1(m_1, f_L)
        h_2 = self.hfs2(m_2, f_L)
        s = self.sgr(c, h_1, h_2)
        s = F.interpolate(s, scale_factor=2, mode='bilinear', align_corners=True)
        o_2 = self.out_2(s)
        o_1 = self.rfm(m_1, s)
        o_1 = F.interpolate(o_1, scale_factor=4, mode='bilinear', align_corners=True)
        o_2 = F.interpolate(o_2, scale_factor=4, mode='bilinear', align_corners=True)
        return o_1, o_2


if __name__ == "__main__":
    with torch.no_grad(): 
        model = FA_SAM2().cuda()
        x = torch.randn(2, 3, 352, 352).cuda()
        o_1, o_2 = model(x)
        print(o_1.shape)
