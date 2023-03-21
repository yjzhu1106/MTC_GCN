import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn import Parameter
from gcn import GraphConvolution
from utils import gen_A, gen_adj
import torchvision.models as models



class MTC_GCN(nn.Module):
    """
    Based on the code of ML-GCN https://github.com/Megvii-Nanjing/ML-GCN
    """

    def __init__(self, adj_file=None, in_channel=300, input_size=227):
        super(MTC_GCN, self).__init__()

        # todo: 确定一下输出的维度
        # self.features = models.densenet121(pretrained=True).features
        self.features = MultiChannel(filters=2048)
        if input_size == 227:
            self.pooling = nn.MaxPool2d(7, 7)
        else:
            self.pooling = nn.MaxPool2d(3, 3)

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 2048)
        # self.gc3 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        print(self.A)

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        # x = self.relu(x)
        # x = self.gc3(x, adj)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        return x[:, :7], x[:, 7:]


class MultiChannel(nn.Module):
    def __init__(self, in_channels=2048, filters=2048):
        super(MultiChannel, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.feature = nn.Sequential(*list(resnet50.children())[:-2])
        # self.feature = models.resnet50(pretrained=True)
        self.f1_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.f2_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=3, padding=3, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.f3_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.f4_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=7, padding=7, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.sigmoid_activation = nn.Sigmoid()

        self.pool_layer = nn.AvgPool2d(kernel_size=(1, 1))

        self.attention1_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=7, padding=7, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=7, padding=7, bias=False)
        )

        self.attention2_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=7, padding=7, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(in_channels, filters, kernel_size=3, dilation=7, padding=7, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)
        self.last_conv = nn.Conv2d(2048, filters, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inputs):

        x = self.feature(inputs)
        # x = torch.transpose(x, 1, 3)  # 将维度 1 移动到维度 3 的位置
        # x = torch.transpose(x, 1, 2)
        f1 = self.f1_conv(x)
        f2 = self.f2_conv(x)
        f3 = self.f3_conv(x)
        f4 = self.f4_conv(x)

        fb1 = torch.add(f1,f2)
        fb2 = torch.add(f3 , f4)

        wb1 = self.sigmoid_activation(fb1)
        wb2 = self.sigmoid_activation(fb2)

        fr1 = torch.mul(fb1, wb1)
        fr2 = torch.mul(fb2, wb2)

        yc = self.pool_layer(fr1)
        zc = self.pool_layer(fr2)

        p1 = self.attention1_conv(yc)
        p2 = self.attention2_conv(zc)

        p1 = self.softmax(p1)
        p2 = self.softmax(p2)

        fa1 = torch.add(torch.mul(f1, p1), torch.mul(f2, p1))
        fa2 = torch.add(torch.mul(f3, p2), torch.mul(f4, p2))

        fa = torch.add(fa1, fa2)

        ffus = self.last_conv(fa)

        return ffus



