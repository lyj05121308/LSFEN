
import torch
import torch.nn as nn


class AM(nn.Module):
    def __init__(self,feature_dim,channel):
        super(AM, self).__init__()
        self.aver_pool = nn.AvgPool1d(kernel_size=feature_dim)
        self.conv1 = nn.Conv1d(in_channels=channel,out_channels=(channel//8),kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=(channel//8), out_channels=channel,kernel_size=1)
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        out = self.aver_pool(x)
        B,C,H = out.shape
        # out = out.reshape(B,-1)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmod(out)
        out = out.reshape(B,C,-1)
        # out = self.softmax(out)
        out = out*x

        return out


if __name__ == '__main__':
    a = torch.rand((50,64,1024)).cuda()
    model = AM(feature_dim=1024,channel=64).cuda()
    se_out = model(a)
    print(se_out.size())