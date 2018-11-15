######################################################## Basic Unets
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class outconv_without_sigmoid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv_without_sigmoid, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet_light(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_light, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = outconv(16, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet_heavy(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_heavy, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

######################################################## Unet with multiple inputs (with different resolutions)
class UNet_multi(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_multi, self).__init__()
        self.inc1 = inconv(n_channels, 8)
        self.down11 = down(8, 16)
        self.down12 = down(16, 32)
        self.down13 = down(32, 64)
        self.down14 = down(64, 64)
        self.up11 = up(128, 32)
        self.up12 = up(64, 16)
        self.up13 = up(32, 8)
        self.up14 = up(16, 8)
        self.up15 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.up16 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.outc_without_sigmoid1 = outconv_without_sigmoid(8, 8)
        
        self.inc2 = inconv(n_channels, 8)
        self.down21 = down(8, 16)
        self.down22 = down(16, 32)
        self.down23 = down(32, 64)
        self.down24 = down(64, 64)
        self.up21 = up(128, 32)
        self.up22 = up(64, 16)
        self.up23 = up(32, 8)
        self.up24 = up(16, 8)
        self.up25 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.outc_without_sigmoid2 = outconv_without_sigmoid(8, 8)
        
        self.inc3 = inconv(n_channels, 8)
        self.down31 = down(8, 16)
        self.down32 = down(16, 32)
        self.down33 = down(32, 64)
        self.down34 = down(64, 64)
        self.up31 = up(128, 32)
        self.up32 = up(64, 16)
        self.up33 = up(32, 8)
        self.up34 = up(16, 8)
        self.outc_without_sigmoid3 = outconv_without_sigmoid(8, 8)
        
        self.outc1 = outconv(24, 8)
        self.outc2 = outconv(8, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_big_path(self, x):
        x1 = self.inc3(x)
        x2 = self.down31(x1)
        x3 = self.down32(x2)
        x4 = self.down33(x3)
        x5 = self.down34(x4)
        x = self.up31(x5, x4)
        x = self.up32(x, x3)
        x = self.up33(x, x2)
        x = self.up34(x, x1)
        x = self.outc_without_sigmoid3(x)
        return x

    def forward_middle_path(self, x):
        x1 = self.inc2(x)
        x2 = self.down21(x1)
        x3 = self.down22(x2)
        x4 = self.down23(x3)
        x5 = self.down24(x4)
        x = self.up21(x5, x4)
        x = self.up22(x, x3)
        x = self.up23(x, x2)
        x = self.up24(x, x1)
        x = self.up25(x)
        x = self.outc_without_sigmoid2(x)
        return x

    def forward_small_path(self, x):
        x1 = self.inc1(x)
        x2 = self.down11(x1)
        x3 = self.down12(x2)
        x4 = self.down13(x3)
        x5 = self.down14(x4)
        x = self.up11(x5, x4)
        x = self.up12(x, x3)
        x = self.up13(x, x2)
        x = self.up14(x, x1)
        x = self.up15(x)
        x = self.up16(x)
        x = self.outc_without_sigmoid1(x)
        return x

    def forward(self, features1, features2, features3):
        x1 = self.forward_small_path(features1)
        x2 = self.forward_middle_path(features2)
        x3 = self.forward_big_path(features3)
        x = torch.cat([x1,x2,x3], 1)
        x = self.outc1(x)
        x = self.outc2(x)
        return x
    
######################################################## VGG16
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class UNet_VGG16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32):
        super(UNet_VGG16, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.vgg11(pretrained=True).features
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final_relu = nn.ReLU(inplace=True)
        self.final2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.sigmoid(self.final2(self.final(dec1)))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
    
########################################
class ConvNoPooling(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch, kernel_size):
        super(ConvNoPooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ConvPooling(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch, kernel_size):
        super(ConvPooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class MultiResolutionClassifier(nn.Module):
    def __init__(self, num_classes = 1, num_channels = 1):
        super(MultiResolutionClassifier, self).__init__()
        
        self.conv11 = ConvPooling(num_channels, 24, 5)
        self.conv12 = ConvPooling(24, 32, 3)
        self.conv13 = ConvNoPooling(32, 48, 3)
        self.flatten1 = torch.nn.Linear(3*3*48, 256)
        
        self.conv21 = ConvPooling(num_channels, 24, 7)
        self.conv22 = ConvPooling(24, 32, 5)
        self.conv23 = ConvPooling(32, 48, 3)
        self.flatten2 = torch.nn.Linear(5*5*48, 256)
        
        self.conv31 = ConvPooling(num_channels, 24, 9)
        self.conv32 = ConvPooling(24, 32, 7)
        self.conv33 = ConvPooling(32, 48, 5)
        self.flatten3 = torch.nn.Linear(6*6*48, 256)
        
        self.outc = torch.nn.Linear(256*3, num_classes)
        if num_classes == 1:
            self.out_softmax = torch.nn.Sigmoid()
        else:
            self.out_softmax = torch.nn.Softmax()
        
    def forward_small_path(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = x.view(x.shape[0], -1)
        x = self.flatten1(x)
        return x
    
    def forward_middle_path(self, x):
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = x.view(x.shape[0], -1)
        x = self.flatten2(x)
        return x
    
    def forward_big_path(self, x):
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = x.view(x.shape[0], -1)
        x = self.flatten3(x)
        return x
        
    def forward(self, patch1, patch2, patch3):
        x1 = self.forward_small_path(patch1)
        x2 = self.forward_middle_path(patch2)
        x3 = self.forward_big_path(patch3)
        x = torch.cat([x1,x2,x3], 1)
        x = self.outc(x)
        x = self.out_softmax(x)
        return x
     