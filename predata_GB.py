import random
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from keras import utils
import os
import torch
import pandas as pd


def window_sample(length,step,data):
    sample_num = 0
    data_real = []
    num = data.size
    # print(num)
    while length>0:
        data_i=[]
        for i in range(length):
            data_i.append(data[sample_num*step+i+4])
        # 由于数据的原因，从第五个数据开始采样
        # 相应的下面((data.size-length-4)/step+1)这里需要减4

        sample_num += 1
        data_real.append(data_i)
        if sample_num == int((data.size-length-4)/step+1):
            break
        # print(sample_num)
    data_real = np.array(data_real)
    return data_real

def deal_data(data, length, label,step):
    """
    对数据进行处理，返回带有标签数据
    :param data:
    :param length:
    :param label:标签值
    :return:返回处理后带标签的数据集
    """

    data = np.reshape(data, (-1))
    data = window_sample(length,step,data)
    num = data.shape[0]
    # print(data.shape)
    # print(num)

    min_max_scaler = preprocessing.MinMaxScaler()
    # 将数据的每一个特征缩放到给定的范围，将数据的每一个属性值减去其最小值，然后除以其极差（最大值 - 最小值）
    data = min_max_scaler.fit_transform(data)

    label = np.ones((num,1))*label
    # 创建n*1的数组
    return np.column_stack((data,label))
#     列合并 数据与标签合并

def other_label(data,label):
    num = data.shape[0]
    label = np.ones((num,1))*label
    return  np.column_stack((data,label))



def open_data(bath_path, key_num, channel):
    """

    :param bath_path: 文件的位置
    :param key_num: 文件名称
    :param channel: 打开的文件第几个通道作为实验数据
    :return:
    """
    name = str(key_num) + '.csv'
    file_path = os.path.join(bath_path, name)
    # 找到文件路径
    data_first= pd.read_csv(file_path,skiprows=16,header=None)
    data = data_first.values[:,channel-1]
    return data

    # return data

def split_data(data,split_rate):
    """
    将数据集进行分类，分为train、eval、test
    :param data:带有标签的数据集
    :param split_rate:
    :return:返回拆分后的数据集
    """
    length = len(data)
    # 数据行的大小
    num1 = int(length*split_rate[0])
    num2 = int(length*split_rate[1])

    index1 = random.sample(range(num1), num1)
    # 数据打乱，选择num1个数
    train = data[index1]
    data = np.delete(data,index1,axis=0)
    # 按行删除数据
    index2 = random.sample(range(num2),num2)
    eval = data[index2]
    test = np.delete(data,index2,axis=0)
    return train, eval, test


def select_data(data, num):
    index = random.sample(range(len(data)), (int(num*5)))
    data = data[index]
    return data

def load_data(ratio,channel1,step,scenes,length = 1024,hp = [0,1,2],fault_num = [2, 4, 6],split_rate = [0.7,0.1,0.2]):
    # 记得修改augmentation中数据的1000

    '''
    返回训练所用数据集以及测试所用数据集
    :param num: 选取的类的样本个数
    :param length: 每个样本长度
    :param hp: 叠加数
    :param fault_num:代表故障数字
    :param split_rate: 划分比例
    :return: 返回训练集集以及测试集
    '''
    num = [200, (200/ratio)]

    # 齿轮及轴承数据的保存路径
    bath_path1 = r".\bearingset"
    bath_path2 = r".\gearset"
    if scenes == 0:
        LT_opt = [1.5,1]
    if scenes == 1:
        LT_opt = [2, 1]
    if scenes == 2:
        LT_opt = [1,1]

    data_list = []
    label = 0
    data_n = []


    data_terminal0 = []
    # 健康状态
    for i in hp:
        name = f'health_20_0'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label),step)
        data = other_label(data,label)  #第二次任务数标签
        data = other_label(data, label)  #第三次任务标签
        for i in data:
            data_terminal0.append(i.tolist())

    data_terminal = np.array(data_terminal0)
    normal_data = select_data(data_terminal, num[0])
    data_list.append(normal_data)
    # print(data_list[0])


    data_terminal1 = []
    # fault1滚子故障
    for i in hp:
        name = f'ball_20_0'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data,(label+1))  #第二次任务数标签
        data = other_label(data, (label+1))  #第三次任务标签
        for i in data:
            data_terminal1.append(i.tolist())

    data_terminal = np.array(data_terminal1)
    normal_data = select_data(data_terminal, num[1]*LT_opt[0])
    data_list.append(normal_data)


    data_terminal2 = []
    # fualt2 滚子故障
    for i in hp:
        name = f'ball_30_2'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data,(label+1))  #第二次任务数标签
        data = other_label(data, (label+2))  #第三次任务标签

        for i in data:
            data_terminal2.append(i.tolist())

    data_terminal = np.array(data_terminal2)
    normal_data = select_data(data_terminal, num[1]*LT_opt[1])
    data_list.append(normal_data)

    data_terminal3 = []
    # fault3 内圈故障
    for i in hp:
        name = f'inner_20_0'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data,(label+2))  #第二次任务数标签
        data = other_label(data, (label+3))  #第三次任务标签
        for i in data:
            data_terminal3.append(i.tolist())

    data_terminal = np.array(data_terminal3)
    normal_data = select_data(data_terminal, num[1]*LT_opt[0])
    data_list.append(normal_data)


    data_terminal4 = []
    # fault4  内圈故障
    for i in hp:
        name = f'inner_30_2'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data,(label+2))  #第二次任务数标签
        data = other_label(data, (label+4))  #第三次任务标签
        for i in data:
            data_terminal4.append(i.tolist())

    data_terminal = np.array(data_terminal4)
    normal_data = select_data(data_terminal, num[1]*LT_opt[1])
    data_list.append(normal_data)

    data_terminal5 = []
    # fault5  外圈故障
    for i in hp:
        name = f'outer_20_0'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data,(label+3))  #第二次任务数标签
        data = other_label(data, (label+5))  #第三次任务标签
        for i in data:
            data_terminal5.append(i.tolist())

    data_terminal = np.array(data_terminal5)
    normal_data = select_data(data_terminal, num[1]*LT_opt[0])
    data_list.append(normal_data)

    data_terminal6 = []
    # fault6  外圈故障
    for i in hp:
        name = f'outer_30_2'
        data = open_data(bath_path=bath_path1, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 1), step)
        data = other_label(data, (label + 3))  # 第二次任务数标签
        data = other_label(data, (label + 6))  # 第三次任务标签
        for i in data:
            data_terminal6.append(i.tolist())

    data_terminal = np.array(data_terminal6)
    normal_data = select_data(data_terminal, num[1]*LT_opt[1])
    data_list.append(normal_data)

    data_terminal7 = []
    # fault7  齿面缺失
    for i in hp:
        name = f'Chipped_20_0'
        data = open_data(bath_path=bath_path2, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 2), step)
        data = other_label(data, (label + 4))  # 第二次任务数标签
        data = other_label(data, (label + 7))  # 第三次任务标签
        for i in data:
            data_terminal7.append(i.tolist())

    data_terminal = np.array(data_terminal7)
    normal_data = select_data(data_terminal, num[1]*LT_opt[0])
    data_list.append(normal_data)


    data_terminal8 = []
    # fault8  齿面缺失
    for i in hp:
        name = f'Chipped_30_2'
        data = open_data(bath_path=bath_path2, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 2), step)
        data = other_label(data, (label + 4))  # 第二次任务数标签
        data = other_label(data, (label + 8))  # 第三次任务标签
        for i in data:
            data_terminal8.append(i.tolist())

    data_terminal = np.array(data_terminal8)
    normal_data = select_data(data_terminal, num[1]*LT_opt[1])
    data_list.append(normal_data)

    data_terminal9 = []
    # fault9  齿根断裂
    for i in hp:
        name = f'Root_20_0'
        data = open_data(bath_path=bath_path2, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 2), step)
        data = other_label(data, (label + 5))  # 第二次任务数标签
        data = other_label(data, (label + 9))  # 第三次任务标签
        for i in data:
            data_terminal9.append(i.tolist())

    data_terminal = np.array(data_terminal9)
    normal_data = select_data(data_terminal, num[1]*LT_opt[0])
    data_list.append(normal_data)

    data_terminal10 = []
    # fault10  齿根断裂
    for i in hp:
        name = f'Root_30_2'
        data = open_data(bath_path=bath_path2, key_num=name, channel=channel1)
        data = deal_data(data, length, (label + 2), step)
        data = other_label(data, (label + 5))  # 第二次任务数标签
        data = other_label(data, (label + 10))  # 第三次任务标签
        for i in data:
            data_terminal10.append(i.tolist())

    data_terminal = np.array(data_terminal10)
    normal_data = select_data(data_terminal, num[1]*LT_opt[1])
    data_list.append(normal_data)
    train = []
    eval = []
    test = []
    for data in data_list:
        a, b, c = split_data(data,split_rate)
        train.append(a)
        eval.append(b)
        test.append(c)

    pre_train = []
    for i in train:
        for j in i:
            pre_train.append(j)
    # print(pre_train)
    train = np.reshape(pre_train,(-1,length+3))
    train = train[random.sample(range(len(train)), len(train))]

    train_data = train[:,0:length]
    # train_label = utils.to_categorical(train[:,length],(1+2*len(fault_num)))
    train_label1,train_lable2, train_label3 = train[:,length],train[:,length+1],train[:,length+2]

    pre_eval = []
    for i in eval:
        for j in i:
            pre_eval.append(j)
    # print(pre_train)
    eval = np.reshape(pre_eval, (-1, length + 3))
    eval = eval[random.sample(range(len(eval)),len(eval))]
    eval_data = eval[:,0:length]
    # print(eval_data.shape)
    # eval_label = utils.to_categorical(eval[:,length],(1+2*len(fault_num)))
    eval_label1,eval_label2, eval_label3 = eval[:,length],eval[:,length+1],eval[:,length+2]

    pre_test = []
    for i in test:
        for j in i:
            pre_test.append(j)
    # print(pre_train)
    test = np.reshape(pre_test, (-1, length + 3))
    # print(len(test))
    test = test[random.sample(range(len(test)),len(test))]
    test_data = test[:,0:length]
    test_label1,test_label2,test_label3 = test[:,length],test[:,length+1],test[:,length+2]



    return torch.tensor(train_data),torch.tensor(train_label1),torch.tensor(train_lable2),torch.tensor(train_label3),\
           torch.tensor(eval_data),torch.tensor(eval_label1),torch.tensor(eval_label2),torch.tensor(eval_label3),\
           torch.tensor(test_data),torch.tensor(test_label1),torch.tensor(test_label2),torch.tensor(test_label3)
#   其中label表示one-hot编码后的标签值，orlabel表示原来的标签值

if __name__ == '__main__':
    train_data,yt1,yt2,yt3,eval_data,ye1,ye2,ye3,test_data,ys1,ys2,ys3 =load_data(ratio=5,channel1=3, step=1024,length=1024, scenes=0)
    y1_label,y1_num = torch.unique(yt1,return_counts = True)
    y2_label, y2_num = torch.unique(yt2, return_counts=True)
    y3_label, y3_num = torch.unique(yt3, return_counts=True)
    print(y1_label,y1_num)
    print(y2_label,y2_num)
    print(y3_label,y3_num)
    # load_data(ratio=5, channel='GBz')
