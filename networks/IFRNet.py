import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def get_robust_weight(flow_pred, flow_gt, beta):
    epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-beta * epe)
    return robust_weight


class Ternary(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        self.w = self.w.to(tensor.device)
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
        
    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Geometry(nn.Module):
    def __init__(self, patch_size=3):
        super(Geometry, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        self.w = self.w.to(tensor.device)
        b, c, h, w = tensor.size()
        tensor_ = tensor.reshape(b*c, 1, h, w)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c*(self.patch_size**2), h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_ ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss


class Charbonnier_Ada(nn.Module):
    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss
    

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder_L(nn.Module):
    def __init__(self):
        super(Encoder_L, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 64, 7, 2, 3), 
            convrelu(64, 64, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(64, 96, 3, 2, 1), 
            convrelu(96, 96, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(96, 144, 3, 2, 1), 
            convrelu(144, 144, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(144, 192, 3, 2, 1), 
            convrelu(192, 192, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4_L(nn.Module):
    def __init__(self):
        super(Decoder4_L, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(384+1, 384), 
            ResBlock(384, 64), 
            nn.ConvTranspose2d(384, 148, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3_L(nn.Module):
    def __init__(self):
        super(Decoder3_L, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(436, 432), 
            ResBlock(432, 64), 
            nn.ConvTranspose2d(432, 100, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2_L(nn.Module):
    def __init__(self):
        super(Decoder2_L, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(292, 288), 
            ResBlock(288, 64), 
            nn.ConvTranspose2d(288, 68, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1_L(nn.Module):
    def __init__(self):
        super(Decoder1_L, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(196, 192), 
            ResBlock(192, 64), 
            nn.ConvTranspose2d(192, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Encoder_S(nn.Module):
    def __init__(self):
        super(Encoder_S, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 24, 3, 2, 1), 
            convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(24, 36, 3, 2, 1), 
            convrelu(36, 36, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(36, 54, 3, 2, 1), 
            convrelu(54, 54, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(54, 72, 3, 2, 1), 
            convrelu(72, 72, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4_S(nn.Module):
    def __init__(self):
        super(Decoder4_S, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(144+1, 144), 
            ResBlock(144, 24), 
            nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3_S(nn.Module):
    def __init__(self):
        super(Decoder3_S, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(166, 162), 
            ResBlock(162, 24), 
            nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2_S(nn.Module):
    def __init__(self):
        super(Decoder2_S, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(112, 108), 
            ResBlock(108, 24), 
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1_S(nn.Module):
    def __init__(self):
        super(Decoder1_S, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(76, 72), 
            ResBlock(72, 24), 
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out
    

class IFRNet(nn.Module):
    def __init__(self, scale="large"):
        super(IFRNet, self).__init__()
        if scale == "large":
            self.encoder = Encoder_L()
            self.decoder4 = Decoder4_L()
            self.decoder3 = Decoder3_L()
            self.decoder2 = Decoder2_L()
            self.decoder1 = Decoder1_L()
        if scale == "small":
            self.encoder = Encoder_S()
            self.decoder4 = Decoder4_S()
            self.decoder3 = Decoder3_S()
            self.decoder2 = Decoder2_S()
            self.decoder1 = Decoder1_S()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)
   

    def forward(self, img0, img1, embt, imgt=None, scale_factor=(1.0, 0.5), onlyFlow=False):
        _, _, H, W = img0.shape
        if H == 320 and W == 1024:
            scale_factor = (0.6, 0.3125)

        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        fh, fw = int(H*scale_factor[0]), int(W*scale_factor[1])
        img0_ = F.interpolate(img0, size=(fh, fw), mode="bilinear", align_corners=False)
        img1_ = F.interpolate(img1, size=(fh, fw), mode="bilinear", align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        if imgt is not None:
            imgt = imgt - mean_
            imgt_ = F.interpolate(imgt, size=(fh, fw), mode="bilinear", align_corners=False)  
            ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        ## we find that the residual component up_res_1 can cause color slanting in KITTI VFI results.
        ## So we drop it out.
        # up_res_1 = out1[:, 5:]

        up_flow0_1 = F.interpolate(up_flow0_1, size=(H, W), mode="bilinear", align_corners=False)
        up_flow0_1[:, 0, :, :] *= (1.0/scale_factor[1])
        up_flow0_1[:, 1, :, :] *= (1.0/scale_factor[0])
        up_flow1_1 = F.interpolate(up_flow1_1, size=(H, W), mode="bilinear", align_corners=False)
        up_flow1_1[:, 0, :, :] *= (1.0/scale_factor[1])
        up_flow1_1[:, 1, :, :] *= (1.0/scale_factor[0])
        up_mask_1 = F.interpolate(up_mask_1, size=(H, W), mode="bilinear", align_corners=False)

        if onlyFlow:
            return up_flow0_1, up_flow1_1, up_mask_1
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp
        # imgt_pred = imgt_merge + up_res_1
        imgt_pred = imgt_merge + mean_
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if imgt is not None:
            loss_rec = self.l1_loss(imgt_merge - imgt) + self.tr_loss(imgt_merge, imgt)
            loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
            loss = loss_rec + loss_geo
            return imgt_pred, loss, up_flow0_1, up_flow1_1, up_mask_1
        else:
            return imgt_pred, up_flow0_1, up_flow1_1, up_mask_1


