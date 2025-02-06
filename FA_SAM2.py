import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from modules import *
from torchvision.transforms.functional import rgb_to_grayscale

class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class CLA(nn.Module):
    def __init__(self, channel):
        super(CLA, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_plus1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_plus2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat9 = BasicConv2d(2*channel, channel, 3, padding=1)
        self.conv_concat8 = BasicConv2d(2*channel, channel, 3, padding=1)


        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x2_1 = self.conv_plus1(self.conv_upsample1(self.upsample(x1)) * x2 + x2)
        x1_up2 =self.conv_upsample9(self.upsample(x1))
        x21 = self.conv_concat9(torch.cat((x2_1,x1_up2),dim=1))

        x3_2 = self.conv_plus2(self.conv_upsample2(self.upsample(x2)) * x3 + x3)
        x12_up2 = self.conv_upsample8(self.upsample(x21))
        x32 = self.conv_concat8(torch.cat((x3_2,x12_up2),dim=1))

        return x32

class SGR(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()

        self.pwconv = ConvBNReLU(in_channel=in_chs, out_channel=out_chs, kernel_size=1)
        self.se = SELayer(channel=out_chs, reduction=4)

    def forward(self, x):
        out = self.se(self.pwconv(x))

        return out

class SEA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(SEA, self).__init__()
        out_channels = int(channels // r)

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # channel_att
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, kernel_size=1),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        # self.channel_att = SELayer()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # local_att
        xl = self.local_att(x)
        # global_att
        xg = self.global_att(x)
        # channel_att
        xc = self.channel_att(x)
        # weighted sum of local and global attention features
        xlg = xl + xg
        # apply channel attention
        xla = xc * xlg
        # sigmoid activation
        wei = self.sig(xla)

        return wei

def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=True)

class HFS(nn.Module):
    def __init__(self, channels=64):
        super(HFS, self).__init__()

        self.sea = SEA(channels)
        self.upsample = cus_sample
        self.conv_xy = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_xy = nn.BatchNorm2d(channels * 2)
        self.conv_gate = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels * 2)
        self.sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_feature):
        xsize = x.size()[2:]
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        x_edge = x + x * edge_input
        xy = torch.cat((x, x_edge), dim=1)

        xy_conv = self.conv_xy(xy)
        xy_bn = self.bn_xy(xy_conv)
        xy_relu = self.relu(xy_bn)

        gate_conv = self.conv_gate(xy)
        gate_bn = self.bn_gate(gate_conv)
        gate_sigmoid = self.sigmoid(gate_bn)

        feat = xy_relu * gate_sigmoid
        feat = self.conv_out(feat)

        feat_weight = self.sea(feat)

        feat_weighted_x = x * feat_weight
        feat_weighted_y = x_edge * (1 - feat_weight)

        feat_sum = feat_weighted_x + feat_weighted_y

        return feat_sum
    
class Decoder(nn.Module):
    def __init__(self, channels=[64, 64, 64, 64]): #[64, 128, 320, 512]
        super().__init__()
        self.agg = CLA(channels[0])

        self.hfs1 = HFS(channels[0])
        self.hfs2 = HFS(channels[1])

        self.fuse_x2 = SGR(in_chs=2 * channels[1], out_chs=channels[1])

        self.clear1_down = Down(ch_in=channels[0], ch_out=channels[1])

    def forward(self, x1, x2, x3, x4, edge_feature):
        contour2 = self.agg(x4, x3, x2)

        edge1 = self.hfs1(x1, edge_feature)
        edge2 = self.hfs2(x2, edge_feature)

        edge1_down_x2 = self.clear1_down(edge1)

        clear_feature1 = torch.cat((contour2 + edge1_down_x2, contour2 + edge2), dim=1)
        clear_feature1 = self.fuse_x2(clear_feature1)

        return clear_feature1 
    
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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MSE(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(MSE, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            BasicConv2d(in_channel, out_channel, 1)
        )
        
        self.conv_cat = BasicConv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class FA_SAM2(nn.Module):
    def __init__(self, checkpoint_path) -> None:
        super(FA_SAM2, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint_path)
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

        self.decoder = Decoder(channels=[64, 64, 64, 64]) #144, 288, 576, 1152

        self.feat_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.feat_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.feat_conv3 = BasicConv2d(64, 64, kernel_size=1)
        self.feat_conv4 = BasicConv2d(64, 1, kernel_size=1)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        freq_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        freq_feature = freq_feature[1]

        x1, x2, x3, x4 = self.encoder(x)
        x_1, x_2, x_3, x_4 = self.mse1(x1), self.mse2(x2), self.mse3(x3), self.mse4(x4)

        feat_map = self.decoder(x_1, x_2, x_3, x_4, freq_feature) 
        feat_map = F.interpolate(feat_map, scale_factor=2, mode='bilinear', align_corners=True)  
        
        out_map = self.out(feat_map)
        x = -1*(torch.sigmoid(out_map)) + 1
        x = x.expand(-1, 64, -1, -1).mul(x_1)
        x = self.feat_conv1(x)
        x = F.relu(self.feat_conv2(x))
        x = F.relu(self.feat_conv3(x))
        x = x + feat_map
        x = self.feat_conv4(x)

        out = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        out_map = F.interpolate(out_map, scale_factor=4, mode='bilinear', align_corners=True)
        return out, out_map

if __name__ == "__main__":
    with torch.no_grad():
        checkpoint_path = "/root/autodl-tmp/sam2_hiera_large.pt" 
        model = FA_SAM2(checkpoint_path).cuda()
        x = torch.randn(2, 3, 352, 352).cuda()
        out, feat_map = model(x)
        print(out.shape, feat_map.shape)
