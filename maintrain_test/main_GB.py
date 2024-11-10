import torch
import torchvision
import torch.nn as nn
import self_dataset
# 数据集加载
import torch.utils.data as Data
import predata_GB
# 数据预处理
import torch.nn.functional as F
from model import batchcost_loss
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score
from model import based_transformer
from model import cross_tranformer
from model import pareto_weight
from model import CFE_net
from model import short_cnn
from sklearn.metrics import confusion_matrix
import time
from torch.utils.data import Dataset, DataLoader



def train(train_data,yt1,yt2,yt3,eval_data,ye1,ye2,ye3,ratio,drop_ratio1,drop_ratio2,heads,scene):
    length = 1024
    model_cnn = short_cnn.sfeasure_exactor(length=1024,resnet_in_channle=1,resnet_out_channel_1=64,
                         resnet_out_channel_2=128,resnet_out_channel_3=128,resnet_out_channel_4=192,group_number=8).cuda()

    # 记住此处的100也得修改
    model_task1 = based_transformer.based_transformer(origin_channel=192
        ,embedding_channel=128,feature=512,
        patch_size=8,
        num_classes = 3,
        dim =128,
        depth = 2,
        heads = heads,
        mlp_dim =256,
        dropout = 0.2,
        emb_dropout = 0.2, pool='mean'
    ).cuda()
    model_task2 = cross_tranformer.cross_transformer(origin_channel=192
        ,embedding_channel=128,feature=512,
        patch_size=8,
        num_classes = 6,
        dim = 128,
        depth = 2,
        heads = heads,
        mlp_dim =256,
        dropout = drop_ratio1,
        emb_dropout = drop_ratio1,pool='mean'


    ).cuda()
    model_task3 = cross_tranformer.cross_transformer(origin_channel=192
        ,embedding_channel=128,feature=512,
        patch_size=8,
        num_classes = 11,
        dim = 128,
        depth = 2,
        heads = heads,
        mlp_dim = 256,
        dropout = drop_ratio2,
        emb_dropout =drop_ratio2,pool='mean'
    ).cuda()
    am = CFE_net.AM(feature_dim=512,channel=192).cuda()
    Epoch =50

    batch_size = 64
    # train_data = Increase_dim.incre_dim(train_data)
    lr = 0.0001
    # print(train_data.shape)
    B1,C1 = train_data.shape
    train_data = train_data.reshape(B1,1,-1)
    # print(B1)
    B2,C2 = eval_data.shape
    eval_data = eval_data.reshape(B2,1,-1)

    # train_orlabel.view(1,-1)

    loss_origin = batchcost_loss.CostSenLoss()
    # loss_origin = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model_cnn.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model_task1.parameters(), lr=lr)
    optimizer3 = torch.optim.Adam(model_task2.parameters(), lr=lr)
    optimizer4 = torch.optim.Adam(model_task3.parameters(), lr=lr)
    optimizer5 = torch.optim.Adam(am.parameters(), lr=lr)

    for params in model_cnn.parameters():
        params.requires_grad = True

    for params in model_task1.parameters():
        params.requires_grad = True

    for params in model_task2.parameters():
        params.requires_grad = True

    for params in model_task3.parameters():
        params.requires_grad = True

    for params in am.parameters():
        params.requires_grad = True
    torch.set_grad_enabled(True)

    train_alldata = self_dataset.selfdataset(train_data,yt1,yt2,yt3)
    all_number = train_data.size(0)
    val_number = eval_data.size(0)
    train_loader = Data.DataLoader(train_alldata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    # writer = SummaryWriter('logs110')
    step_real = 0
    for i in train_loader:
        step_real+=1
    print(step_real)
    dropratio = '  '.join([str(drop_ratio1), str(drop_ratio2),f'heads = {heads}','with am'])
    title = '  '.join(['epoch','loss_avg','acc_avg'])
    # f.writelines(dropratio+'\n')
    # f.writelines(title+'\n')
    for epoch in range(Epoch):
        running_loss = 0
        acc1 = 0
        acc2 = 0
        acc3 = 0
        all_num = 0
        model_cnn.train()
        model_task1.train()
        model_task2.train()
        model_task3.train()
        w = [0.33,0.33,0.34]
        # w = w.cuda()
        # print(type(w))
        c = [0.00, 0.00, 0.00]
        # print(w1,w2,w3)
        for step, data in enumerate(train_loader):
            x, y1,y2,y3 = data
            # print(type(x))
            #type x  tensor
            x = x.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
            y3 = y3.cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            # print("label.shape:",label.shape)
            # print(label)
            feature1,feature2,feature3= model_cnn(x.to(torch.float32))
            feature3 = am(feature3)
            # print(feature3.size())
            logit1,cross_data1 = model_task1(feature3)
            logit2,cross_data2 = model_task2(feature3,cross_data1)
            logit3,cross_data3 = model_task3(feature3,cross_data2)

            # 分类器输入向量不经过CA
            # feature = Increase_dim.incre_dim(feature)
            pred1 = logit1.argmax(dim=1)
            pred2 = logit2.argmax(dim=1)
            pred3 = logit3.argmax(dim=1)

            w1 = float(w[0])
            w2 = w[1]
            w3 = w[2]

            loss_task1 = loss_origin(logit1,y1.long())
            loss_task2 = loss_origin(logit2,y2.long())
            loss_task3 = loss_origin(logit3,y3.long())
            loss = w1*loss_task1+w2*loss_task2+ w3*loss_task3

            params_task1 = model_task1.parameters()
            params_task2 = model_task2.parameters()
            params_task3 = model_task3.parameters()

            grad_task1 = torch.autograd.grad(loss_task1, params_task1, retain_graph=True)
            grad_task2 = torch.autograd.grad(loss_task2, params_task2, retain_graph=True)
            grad_task3 = torch.autograd.grad(loss_task3, params_task3, retain_graph=True)

            grad_task1 = torch.cat([g.flatten() for g in grad_task1]).cpu()  # 确保在CPU上
            grad_task2 = torch.cat([g.flatten() for g in grad_task2]).cpu()  # 确保在CPU上
            grad_task3 = torch.cat([g.flatten() for g in grad_task3]).cpu()  # 确保在CPU上
            max_len = max(len(grad_task1), len(grad_task2), len(grad_task3))
            grad_task1 = F.pad(grad_task1, (0, max_len - len(grad_task1)))
            grad_task2 = F.pad(grad_task2, (0, max_len - len(grad_task2)))
            grad_task3 = F.pad(grad_task3, (0, max_len - len(grad_task3)))
            stacked_horizontal = np.vstack((grad_task1.numpy(), grad_task2.numpy(), grad_task3.numpy()))
            w = pareto_weight.pareto_efficient_weights(prev_w=np.asarray(w),
                                                           c = np.asarray(c),
                                                           G= stacked_horizontal)

            w =w.tolist()

            loss.backward()

            # print(type(w))
            acc1 += (pred1.data.cuda() == y1.data).sum()
            acc2 += (pred2.data.cuda() == y2.data).sum()
            acc3 += (pred3.data.cuda() == y3.data).sum()
            running_loss += float(loss.data.cpu())
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()


            if ratio ==10:
                # 32      50
                #  16       100
                #  16       1024
                if (step ==(step_real-1)):
                    loss_avg = running_loss/(step+1)
                    # acc_avg = float(acc / all_num)
                    acc_avg1 = float(acc1 / all_number)
                    acc_avg2 = float(acc2 / all_number)
                    acc_avg3 = float(acc3 / all_number)
                    print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg,
                          '|Acc1_avg:%.4f' % acc_avg1,
                          '|Acc2_avg:%.4f' % acc_avg2, '|Acc1_avg3:%.4f' % acc_avg3, )

            if ratio ==20:
                # 26     50
                # 13     100
                # 13     1024
                if(step ==(step_real-1)):
                    loss_avg = running_loss/(step+1)
                    # acc_avg = float(acc / all_num)
                    acc_avg1 = float(acc1 / all_number)
                    acc_avg2 = float(acc2 / all_number)
                    acc_avg3 = float(acc3 / all_number)
                    print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg,
                          '|Acc1_avg:%.4f' % acc_avg1,
                          '|Acc2_avg:%.4f' % acc_avg2, '|Acc1_avg3:%.4f' % acc_avg3, )

            if ratio == 50:
                # 22         50
                # 11           100
                # 11           100
                if (step ==(step_real-1)):
                    loss_avg = running_loss / (step + 1)
                    # acc_avg = float(acc / all_num)
                    acc_avg1 = float(acc1 / all_number)
                    acc_avg2 = float(acc2 / all_number)
                    acc_avg3 = float(acc3 / all_number)
                    print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg,
                          '|Acc1_avg:%.4f' % acc_avg1,
                          '|Acc2_avg:%.4f' % acc_avg2, '|Acc1_avg3:%.4f' % acc_avg3, )

            if ratio ==100:
                # 21         50
                # 10         100
                # 10         1024
                if (step ==(step_real-1)):
                    loss_avg = running_loss/(step+1)
                    # acc_avg = float(acc / all_num)
                    acc_avg1 = float(acc1 / all_number)
                    acc_avg2 = float(acc2 / all_number)
                    acc_avg3 = float(acc3 / all_number)
                    print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg,
                          '|Acc1_avg:%.4f' % acc_avg1,
                          '|Acc2_avg:%.4f' % acc_avg2, '|Acc1_avg3:%.4f' % acc_avg3, )


        model_cnn.eval()
        model_task1.eval()
        model_task2.eval()
        model_task3.eval()
        am.eval()
        val_acc1 = 0  #
        val_acc2 = 0
        val_acc3 = 0
        total_val_loss = 0
        total_val_acc3=0
        eval_alldata = self_dataset.selfdataset(eval_data
                                                , ye1, ye2, ye3)
        eval_loader = Data.DataLoader(eval_alldata, batch_size=batch_size, shuffle=True, num_workers=0,
                                       drop_last=False)
        
        with torch.no_grad():
            for data in eval_loader:
                x,y1,y2,y3 =data
                x = x.cuda()
                y1 = y1.cuda()
                y2=y2.cuda()
                y3 = y3.cuda()
                feature1, feature2, feature3= model_cnn(x.to(torch.float32))
                feature3 = am(feature3)
                logit1, cross_data1 = model_task1(feature3)
                logit2, cross_data2 = model_task2(feature3, cross_data1)
                logit3, cross_data3 = model_task3(feature3, cross_data2)

                pred1 = logit1.argmax(dim=1)
                pred2 = logit2.argmax(dim=1)
                pred3 = logit3.argmax(dim=1)
                loss =loss_origin(logit1, y1.long())+loss_origin(logit2, y2.long())+  loss_origin(logit3, y3.long())
                total_val_loss += loss
                val_acc1 += torch.eq(pred1, y1).float().sum()
                val_acc2 += torch.eq(pred2, y2).float().sum()
                val_acc3 += torch.eq(pred3, y3).float().sum()

            val_loss = total_val_loss/B2
            val_acc1_all = val_acc1/B2
            val_acc2_all = val_acc2/ B2
            val_acc3_all = val_acc3/ B2

            print(f'val_loss:{val_loss}, val_acc1:{val_acc1_all}, val_acc2:{val_acc2_all}, val_acc3:{val_acc3_all}')




    # torch.save(model_cnn, f'model_ESU/model_all_{ratio}_scene{scene}.pkl')
    # torch.save(am,f'model_ESU/am_{ratio}_scene{scene}.pkl')
    # torch.save(model_task1, f'model_ESU/model_task1_{ratio}_scene{scene}.pkl')
    # torch.save(model_task2, f'model_ESU/model_task2_{ratio}_scene{scene}.pkl')
    # torch.save(model_task3, f'model_ESU/model_task3_{ratio}_scene{scene}.pkl')


def test(test_data,ys1,ys2,ys3,ratio,name,scene):
    B1, C1 = test_data.shape
    test_data = test_data.reshape(B1, 1, -1)
    test_alldata = self_dataset.selfdataset(test_data, ys1, ys2, ys3)
    batch_size = 1
    test_loader = Data.DataLoader(test_alldata, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    loss_origin = batchcost_loss.CostSenLoss()

    # 使得每一类的数据都相同

    model_all = torch.load(f'./model_ESU/model_cnn_{ratio}_scene{scene}.pkl').cuda()
    am = torch.load(f'./model_ESU/am_{ratio}_scene{scene}.pkl').cuda()
    model_task1 = torch.load(f'./model_ESU/model_task1_{ratio}_scene{scene}.pkl').cuda()
    model_task2 = torch.load(f'./model_ESU/model_task2_{ratio}_scene{scene}.pkl').cuda()
    model_task3 = torch.load(f'./model_ESU/model_task3_{ratio}_scene{scene}.pkl').cuda()
    torch.set_grad_enabled(False)
    model_all.eval()
    model_task1.eval()
    model_task2.eval()
    model_task3.eval()
    am.eval()
    # length = test_data.data_SQ.size(0)
    acc=0
    running_loss = 0
    plt_data = torch.randn((1,11))
    plt_data = plt_data.cuda()
    # plt_data = plt_data.cuda()
    pre_task1 = []
    pre_task2 = []
    pre_task3 = []
    test_acc1 = 0  # 记录一个epoch中验证集一共预测对了几个
    test_acc2 = 0
    test_acc3 = 0
    total_test_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for data in test_loader:
            x, y1, y2, y3 = data
            x = x.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
            y3 = y3.cuda()
            feature1, feature2, feature3 = model_all(x.to(torch.float32))
            feature3 = am(feature3)
            logit1, cross_data1 = model_task1(feature3)
            logit2, cross_data2 = model_task2(feature3, cross_data1)
            logit3, cross_data3 = model_task3(feature3, cross_data2)
            # print(logit1)
            # logit2 = model_task2(feature2)
            # logit3 = model_task3(feature3)

            # 分类器输入向量不经过CA
            # feature = Increase_dim.incre_dim(feature)
            pred1 = logit1.argmax(dim=1)
            pred2 = logit2.argmax(dim=1)
            pred3 = logit3.argmax(dim=1)
            # print(pred1)
            # print(pred3.size())
            plt_data = torch.cat((plt_data,logit3))
            pre_task1.append(pred1)
            pre_task2.append(pred2)
            pre_task3.append(pred3)
            loss = loss_origin(logit1, y1.long()) + loss_origin(logit2, y2.long()) + loss_origin(logit3, y3.long())
            total_test_loss += loss
            test_acc1 += torch.eq(pred1, y1).float().sum()
            test_acc2 += torch.eq(pred2, y2).float().sum()
            test_acc3 += torch.eq(pred3, y3).float().sum()

        test_loss = total_test_loss / B1
        test_acc1_all = test_acc1 / B1
        test_acc2_all = test_acc2 / B1
        test_acc3_all = test_acc3 / B1
        # print(i)
    end_time = time.time()
    test_time = end_time-start_time
        # print('Predict:', int(pred.data_SQ.cpu()), '|Ground Truth:', int(y.data_SQ))

    pre_task1 = torch.tensor(pre_task1)
    pre_task2 = torch.tensor(pre_task2)
    pre_task3 = torch.tensor(pre_task3)
    # print(pre_task3)
    # print(pre_task1)
    f1_task1 = f1_score(ys1, pre_task1, average='macro')
    f1_task2 = f1_score(ys2, pre_task2, average='macro')
    f1_task3 = f1_score(ys3, pre_task3, average='macro')
    recall_task1 = recall_score(ys1, pre_task1, average='macro')
    recall_task2 = recall_score(ys2, pre_task2, average='macro')
    recall_task3 = recall_score(ys3, pre_task3, average='macro')
    precision_task1 = precision_score(ys1, pre_task1, average='macro')
    precision_task2 = precision_score(ys2, pre_task2, average='macro')
    precision_task3 = precision_score(ys3, pre_task3, average='macro')
    # confusion_matrix_fig(label_true=ys3,label_pre=pre_task3,scene=scene,ratio = ratio)
    print(f'val_loss:{test_loss}, val_acc1:{test_acc1_all}, val_acc2:{test_acc2_all}, val_acc3:{test_acc3_all}')
        # print(plt_data.size())


    return test_acc1_all,test_acc2_all,test_acc3_all,f1_task1,f1_task2,f1_task3,recall_task1,recall_task2,recall_task3,precision_task1,precision_task2,precision_task3,test_time


if __name__ == '__main__':

        # 不同长尾场景
        scene = 0
        # 不同不平衡比
        ratio = 100
        # 模型训练
        train_data, yt1, yt2, yt3, a, b, c, d, e, f, g, h = predata_GB.load_data(ratio=1, channel1=3, step=512,
                                                                                 length=1024, scenes=2)
        # print(train_data.shape)
        a, b, c, d, eval_data, ye1, ye2, ye3, e, f, g, h = predata_GB.load_data(ratio=1, channel1=3, step=512,
                                                                                length=1024, split_rate=[0.8, 0.1, 0.1],
                                                                                scenes=2)
        train(train_data, yt1, yt2, yt3, eval_data, ye1, ye2, ye3, ratio=ratio, drop_ratio1=0.2, drop_ratio2=0.3,
              heads=8, scene=scene)


        # 模型测试
        a, b, c, d, e, f, g, h, test_data, ys1, ys2, ys3 = predata_GB.load_data(ratio=1, channel1=3, step=512,
                                                                                 length=1024,
                                                                                 split_rate=[0.7, 0.1, 0.2], scenes=2)
        acc_1, acc_2, acc_3, f1_1, f1_2, f1_3, re_1, re_2, re_3, pre_1, pre_2, pre_3, test_time = test(test_data, ys1,
                                                                                                       ys2, ys3,
                                                                                                       ratio=ratio,
                                                                                                       name='Proposed',
                                                                                                       scene=scene)
