import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, input_channels, conv1_channels, conv3_channels, conv5_channels, pool_channels):
        super(InceptionBlock, self).__init__()
        # 1x1 卷积层，用于减少参数数量和控制维度
        self.conv1 = nn.Conv2d(input_channels, conv1_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 卷积层
        self.conv3 = nn.Conv2d(input_channels, conv3_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_channels)
        
        # 5x5 卷积层，首先用1x1卷积减少维度
        self.conv5_1 = nn.Conv2d(input_channels, conv5_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5_1 = nn.BatchNorm2d(conv5_channels)
        self.conv5_2 = nn.Conv2d(conv5_channels, conv5_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn5_2 = nn.BatchNorm2d(conv5_channels)
        
        # 3x3 池化层，然后1x1 卷积
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv2d(input_channels, pool_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool_bn = nn.BatchNorm2d(pool_channels)

    def forward(self, x):
        # 1x1 卷积分支
        branch1 = self.relu(self.bn1(self.conv1(x)))
        
        # 3x3 卷积分支
        branch3 = self.relu(self.bn3(self.conv3(x)))
        
        # 5x5 卷积分支
        branch5 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(x))))))

        # 3x3 池化分支
        branch_pool = self.pool(x)
        branch_pool = self.relu(self.pool_bn(self.pool_conv(branch_pool)))
        
        # 合并所有分支
        return torch.cat([branch1, branch3, branch5, branch_pool], 1)

# 假设输入通道数为64
input_channels = 64
inception_block = InceptionBlock(input_channels, 64, 128, 32, 64)

# 创建一个随机张量来模拟输入
x = torch.randn(1, 64, 224, 224)  # 假设输入图像大小为224x224

# 前向传播
output = inception_block(x)
print(output.shape)  # 应该输出合并后的特征图的尺寸
