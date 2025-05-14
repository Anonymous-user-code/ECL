import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#--------------------------LeNet-----------------#
class LeNet(nn.Module):
    name = "LeNet_5"
    in_features = 16 * 4 * 4
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(6, 16, kernel_size=5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     )
        
        self.classifier = nn.Sequential(nn.Linear(16 * 4 * 4, 120),
                                     nn.ReLU(),
                                     nn.Linear(120, 84),
                                     nn.ReLU(),
                                     nn.Linear(84, 10)
                                     )
        
        self.classifier2 = nn.Sequential(nn.Linear(16 * 4 * 4, 120),
                                     nn.ReLU(),
                                     nn.Linear(120, 84),
                                     nn.ReLU(),
                                     nn.Linear(84, 2)
                                     )
    
    def forward(self, x):
        x = self.feature(x)
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
    
    def forward_classifier2(self,x):
        x = self.feature(x)
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier2(x)
        return x
    
    def get_features(self,x):
        x = self.feature(x)
        return x
    
    def classify(self,x):
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
    
#--------------------------ResNet20-----------------#
class BasicBlock(nn.Module):
    expansion = 1
    

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    name = "ResNet20"
    in_features = 16 * 4 * 4
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.feature = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(16),
                                     self._make_layer(16, num_blocks=3, stride=1),
                                     self._make_layer(16, num_blocks=3, stride=2),
                                     self._make_layer(16, num_blocks=3, stride=2),
                                     nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=0, bias=False))
        
        self.classifier = nn.Linear(16 * 4 * 4, 10)
        
        self.classifier2 = nn.Linear(16 * 4 * 4, 2)
        
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.feature(x)
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
    
    def forward_classifier2(self,x):
        x = self.feature(x)
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier2(x)
        return x
    
    def get_features(self,x):
        x = self.feature(x)
        return x
    
    def classify(self,x):
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
    
#--------------------------DenseNet40-----------------#
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        new_feat = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, new_feat], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.avgpool(out)
        return out


class DenseNet40(nn.Module):
    name = "DenseNet40"
    in_features = 16 * 4 * 4

    def __init__(self, num_classes=10):
        super(DenseNet40, self).__init__()
        growth_rate = 1
        
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            DenseBlock(num_layers=12, in_channels=16, growth_rate=growth_rate),
            Transition(in_channels=16 + 12 * growth_rate, out_channels=16),
            DenseBlock(num_layers=12, in_channels=16, growth_rate=growth_rate),
            Transition(in_channels=16 + 12 * growth_rate, out_channels=16),
            DenseBlock(num_layers=12, in_channels=16, growth_rate=growth_rate),
            nn.Conv2d(16 + 12 * growth_rate, 16, kernel_size=4, stride=1, padding=0, bias=False)
        )
        
        self.classifier = nn.Linear(16 * 4 * 4, num_classes),

        self.classifier2 = nn.Linear(16 * 4 * 4, 2)
        
    def forward(self, x):
        x = self.feature(x)  
        x = x.contiguous().view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_classifier2(self,x):
        x = self.feature(x)
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier2(x)
        return x
    
    def get_features(self,x):
        x = self.feature(x)
        return x
    
    def classify(self,x):
        x = x.contiguous().view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x