
import torch
import torch.nn as nn




class short_cnn(nn.Module):
    def __init__(self, in_channel,out_channel1,out_channel2,out_channel3,out_channel4,group_normal,num_classes=14):
        super(short_cnn, self).__init__()
        self.conv1_0 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel1, kernel_size=15, stride=2,padding=7)

        self.relu = nn.ReLU()
        self.gn0 = nn.GroupNorm(group_normal, out_channel1)

        self.conv1_1 = nn.Conv1d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=9, stride=1,padding=4)

        self.relu = nn.ReLU()
        self.gn1 = nn.GroupNorm(group_normal, out_channel2)

        self.conv1_2 = nn.Conv1d(in_channels=out_channel2, out_channels=out_channel3, kernel_size=5, stride=1,padding=2)

        self.relu = nn.ReLU()
        self.gn2 = nn.GroupNorm(group_normal, out_channel3)

        self.conv1_3 = nn.Conv1d(in_channels=out_channel3, out_channels=out_channel4, kernel_size=3, stride=1,padding=1)

        self.relu = nn.ReLU()
        self.gn3 = nn.GroupNorm(group_normal, out_channel4)

        self.shortcut1 = nn.Sequential(
            nn.Conv1d(out_channel1, out_channel2, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channel2)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(out_channel1, out_channel3, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channel3)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv1d(out_channel1, out_channel4, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channel4)
        )


    def forward(self, x):
        out = self.conv1_0(x)
        # out = self.gn0(out)
        out = self.relu(out)
        out = self.gn0(out)
        base_x = out


        out = self.conv1_1(out)
        # out = self.gn1(out)
        out = self.relu(out)
        out = self.gn1(out)
        out1 = out

        out = self.conv1_2(out)
        # out = self.gn2(out)
        out = self.relu(out)
        out = self.gn2(out)
        out2 = out


        out = self.conv1_3(out)
        # out = self.gn3(out)
        out = self.relu(out)
        out = self.gn3(out)
        out3 = out
        # print(out3.size())


        return out1,out2,out3


class sfeasure_exactor(nn.Module):
    def __init__(self,length,resnet_in_channle,resnet_out_channel_1
                 ,resnet_out_channel_2,resnet_out_channel_3,resnet_out_channel_4,group_number):
        super(sfeasure_exactor, self).__init__()
        self.resnet = short_cnn(in_channel=resnet_in_channle,out_channel1=resnet_out_channel_1,out_channel2=resnet_out_channel_2
                                    ,out_channel3=resnet_out_channel_3,out_channel4=resnet_out_channel_4,group_normal=group_number)

    def forward(self,x):
        x1,x2,x3 = self.resnet(x)
        return  x1,x2,x3
        # return x1



if __name__ == '__main__':
    a = torch.rand((32,1,1024)).cuda()
    model  = sfeasure_exactor(length=1024,resnet_in_channle=1,resnet_out_channel_1=64,
                         resnet_out_channel_2=128,resnet_out_channel_3=128,resnet_out_channel_4=192,group_number=32).cuda()
    b,c,d = model(a)
    print(b.size(),
          c.size(),d.size())
