import torch
import torch.nn as nn
import math

class CostSenLoss(nn.Module):
    def __init__(self):
        super(CostSenLoss, self).__init__()

    def forward(self,feature,label):
        min_num = 1e-4
        # print(feature)
        softmax_func = nn.Softmax(dim=1)
        soft_output = softmax_func(feature)
        N_batch = feature.size(0)
        # print(N_batch)
        beta = (1-soft_output)**(2)
        class_name = label.unique()
        class_num = class_name.size(0)
        class_name_list = list(class_name)
        # print(class_name)
        NI = []
        for i in class_name:
            # print(i)
            a = 0
            for j in label:
                if j == i:
                    a+=1
            # print(a)
            NI.append(a)
        # print(NI)
        NI_multi = []
        for i in label:
            a = class_name_list.index(i)
            ni = NI[a]
            ni = torch.tensor(ni)
            alpha = (math.exp((ni/N_batch)-1)/(ni/N_batch))
            NI_multi.append(alpha)
        NI_multi = torch.tensor(NI_multi)
        NI_multi = NI_multi.view(-1,1)
        NI_multi = NI_multi.cuda()
        log_output = torch.log(soft_output)
        # print(log_output)
        cs_feature = NI_multi * beta * log_output
        nllloss_func = nn.NLLLoss()
        cs_loss = nllloss_func(cs_feature, label)
        return cs_loss



if __name__ == '__main__':
    costsen_loss = CostSenLoss()
    x_input = torch.randn(15, 6)  # 随机生成输入
    y_target = torch.tensor([0,1,2,3,4,0,0,1,0,4,2,1,4,3,2])
    y_pre = torch.tensor([0,1,1,3,4,1,0,1,3,4,2,2,4,3,4])
    a = costsen_loss(x_input,y_target)
    print(a)