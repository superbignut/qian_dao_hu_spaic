"""
    在虚拟环境中， 维度开的大一点是没问题的，但是在真实环境中，端到端似乎不太行，

    真正能用到的其实就是，特征提取到的结果，然后用结果来作为spaic的输入才行

    然后的话，打算把情感模型的输出放大为3，分别是，积极，消极，愤怒 三个，对应的动作是 亲近，远离，汪汪叫
"""
import collections
import numpy as np
from tqdm import tqdm
import os
import random
import torch
from SPAIC import spaic
import torch.nn.functional as F
from SPAIC.spaic.Learning.Learner import Learner
from SPAIC.spaic.Library.Network_saver import network_save
from SPAIC.spaic.Library.Network_loader import network_load
from SPAIC.spaic.IO.Dataset import MNIST as dataset
# from SPAIC.spaic.IO.Dataset import CUSTOM_MNIST, NEW_DATA_MNIST
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import csv
import pandas as pd
from collections import deque


EMO = {"NULL":0, "POSITIVE":1, "NEGATIVE":2, "ANGRY":3} # NULL, 积极，消极，愤怒

log_path = './log/ysc'
writer = SummaryWriter(log_path)

# tensorboard.exe --logdir=./log/ysc

root = './SPAIC/spaic/Datasets/MNIST'

model_path = 'save/ysc_model'

buffer_path = 'ysc_buffer.pth'

device = torch.device("cuda:0")

input_node_num = 16
input_num_mul = 16 #  把输入维度放大10倍

input_node_num = input_node_num * input_num_mul #  把输入维度放大10倍

output_node_num = 4

label_num = 100 # 这里要不了这么多

bat_size = 1

backend = spaic.Torch_Backend(device)
backend.dt = 0.1

run_time = 256 * backend.dt 

time_step = int(run_time / backend.dt)

# lr = 0.001 # stdp暂时不用学习率， 真实的学习率应该是体现在算法里面了

im = None

def ysc_create_data_before_pretrain_new_new():
    rows = 10000
    data = []
    groups = {
        'ANGRY': [15, 11],
        'NEGATIVE': [14, 10, 7, 6, 5, 3, 1], # 这里的一个很大的假设就是,如果一起训练可以消极, 那么单个的输入给进来的时候,希望也是消极的!!!!!!
        'POSITIVE': [2, 9, 10, 13],
        'NULL': [0, 4, 8, 12]
        }
    for _ in range(rows):
        selected_group = random.choice(list(groups.values()))
        result_list = [0.0] * input_node_num
        for i in range(input_node_num):
            if i in selected_group:
                result_list[i] = 1.0
            else:
                result_list[i] = random.uniform(0, 0.2)
        data.append(result_list)
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")
def ysc_create_data_after_pretrain_new_new():
    rows = 200
    data = []
    groups = {
        'ANGRY': [15, 11],
        'NEGATIVE': [14, 7, 6, 5, 3, 1],
        'POSITIVE': [2, 9, 10, 13],
        'NULL': [0, 4, 8, 12]
        }
    for _ in range(rows):
        selected_group = random.choice(list(groups.values()))
        result_list = [0.0] * input_node_num
        for i in range(input_node_num):
            if i in selected_group:
                result_list[i] = 1.0 # 测试的话, 可以引入一些噪声
            else:
                result_list[i] = random.uniform(0, 0.2)
            data.append(result_list)
    with open('testdata.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")
def ysc_create_data_before_pretrain_new():
    rows = 10000
    data = []
    for _ in range(rows):
        row = []
        group = [0] * input_node_num
        index = random.randint(0, input_node_num - 1)  # 随机选择一个索引 # 如果大模型非常稀疏的话，可能还要考虑 训练集稀疏一点
        group[index] = 1
        row.extend(group)  # 将组添加到行中
        data.append(row)
    
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")


def ysc_create_data_before_pretrain():
    # 0 1 2 3    4 5   6 7 8   9 10 11 12  训练优先级递增
    #  颜色       酒     imu    大模型评价
    #  4          2      3      4 
    group_sizes = [4, 4, 4, 4]
    rows = 10000
    data = []
    for _ in range(rows):
        row = []
        for size in group_sizes:
            # 创建一个组，随机选择一个位置放置1，其余为0
            group = [0] * size
            index = random.randint(0, size - 1)  # 随机选择一个索引 # 如果大模型非常稀疏的话，可能还要考虑 训练集稀疏一点
            group[index] = 1
            row.extend(group)  # 将组添加到行中
        data.append(row)
    
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")
def ysc_create_data_after_pretrain():
    # 与 ysc_create_data_before_pretrain 先比， 这个就是测试的数据集， 暂时先不添加噪声信号
    group_sizes = [4, 2, 3, 4]
    rows = 100
    data = []
    for _ in range(rows):
        row = []
        for size in group_sizes:
            # 创建一个组，随机选择一个位置放置1，其余为0
            group = [0] * size
            index = random.randint(0, size - 1)  # 随机选择一个索引 # 如果大模型非常稀疏的话，可能还要考虑 训练集稀疏一点
            group[index] = 1
            row.extend(group)  # 将组添加到行中
        data.append(row)
    
    with open('testdata.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")


class YscNet(spaic.Network):
    def __init__(self):
        super().__init__()

        self.input = spaic.Encoder(num=input_node_num, time=run_time, coding_method='poisson', unit_conversion=0.8) # 就是给发放速率乘了因子,from 论文

        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex') # stdp_ex 比 stdp_ih 多了一层阈值的衰减 \tao_\theta, 论文中有提到这个自适应的阈值
        
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih') # 维度都是100

        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # layer1作为了输出层, 兴奋次数作为输出

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=np.random.rand(label_num, input_node_num) * 0.3) # 100 * 784 # 这里其实可以给的高一点， 反正会抑制下去
        
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=np.diag(np.ones(label_num)) * 22.5 ) # 这里
        
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=( np.ones((label_num, label_num)) - np.diag(np.ones(label_num)) ) * (-120)) # 起到一个抑制的作用，除了1-1的前一个，侧向抑制，并起到竞争的效果

        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变

        # self.reward = spaic.Reward(num=label_num, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=1) # 采样频率是每个时间步一次
        #
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        
        self.set_backend(backend)

        self.buffer = [[] for _ in range(output_node_num)] # 0, 1 投票神经元的buffer # 这里不能写成 [[]] * 4 的形式， 否则会出问题

        self.assign_label = None # 统计结束的100 个神经元的代表的情感对象是什么

    def step(self, data, reward=1): # reward 要想加进去的话, 需要去修改一下stdp 的算法

        self.input(data) # 输入数据

        # self.reward(reward) # 这里1，rstdp 退化为stdp 输入奖励 0 则不更新权重

        self.run(run_time) # 前向传播

        return self.output.predict # 输出 结果
    
    def ysc_pre_train_over_save(self):
        # print(self.buffer)
        self.save_state(filename = model_path) # 这里需要手动删除保存的文件夹
        torch.save(self.buffer, buffer_path) # buffer 也需要保存起来

    def new_check_label_from_data(self, data):
        """
            这里暗含了优先级的概念在里面
        """
        if data[0][15] == 1 or data[0][11] == 1:
            return EMO["ANGRY"] # 
        
        elif data[0][14] == 1 or data[0][7] == 1 or data[0][6] == 1 or data[0][5] == 1 or data[0][3] == 1 or data[0][1] == 1:
            return EMO['NEGATIVE']
        
        elif data[0][2] == 1 or data[0][9] == 1 or data[0][10] == 1 or data[0][13] == 1:
            return EMO['POSITIVE']
        
        else:
            return EMO['NULL'] 
        
    def check_label_from_data():
        """ 
            根据1维的数据，返回应有的标签 
            # 0   1   2   3   |    4   5   6   7     |    8   9   10   11    |    12    13     14    15    训练优先级递增
            # 无  红  蓝  暗   |   无   酒  酒  酒    |    无  摸  踢   摸     |    无   积极   批评   侮辱 
            # 
            # 返回 0 1 2 3   代表 NULL, 积极, 消极, 愤怒

            侮辱：
                15 11

            消极：
                14 10 7 6 5 3 1

            积极：
                2 9 10 13

            NULL:
                0 4 8 12


            竟然是预训练就搞的 最简单， 剩下的都留给微调去做. 现在是如果只有 1个是 1 其余全是0的话,会出问题,导致脉冲几乎没有,

            可不可以让 同类的全都一起发放, 比如 侮辱的, 就全都 一起输入进去, 积极的全部一起输入进去

            然后把所有的输入都扩大到 4维 凑成 4 * 4, 即使没有意义, 也都ok, 说不定可以以后进行扩充

            或者说, 多个输入都带有一个信号, 也没有人规定 一个输入 就只能有一个, 比如某些重要的信息, 我就可以构造多个输入进去吗????

            因为在训练的时候 有那么多的不确定性, 多个输入,总也能保证 有大概率是对的

            然后相机其实还可以提取一个信号, 即使 视角如果像素一直不变, 就可以认定没有人,如果变化很大, 则认为 作为有人

            先用 4 * 4 的试一下, 行的话, 甚至可以使用 5 * 5 
        """
        if data[0][12] == 1:
            return EMO["ANGRY"]
        
        elif data[0][11] == 1:
            return EMO['NEGATIVE']
        
        elif data[0][10] == 1:
            return EMO['POSITIVE']
        
        elif data[0][8] == 1:
            return EMO['ANGRY']
        
        elif data[0][7] == 1:
            return EMO['POSITIVE']
        
        elif data[0][5] == 1:
            return EMO['NEGATIVE']
        
        elif data[0][3] == 1:
            return EMO['NULL'] # 这里暂时使用的是NULL， 但是 可以改为愤怒 或者 恐惧 都可以
        
        elif data[0][2] == 1:
            return EMO['POSITIVE']
        
        elif data[0][1] == 1:
            return EMO['NEGATIVE']
        
        else:
            return EMO["NULL"]

    def ysc_pretrain_step(self, data, label=None):
        # 根据与训练数据拿到标签
        # 保存到buffer中
        output = self.step(data)
        print(output)
        label = self.new_check_label_from_data(data)
        # print(label)
        # print(self.buffer)
        # print(output.shape)
        self.buffer[label].append(output)
        # print(self.buffer)
        # print(label, " buffer len is ",len(self.buffer[label]))
        return output

    def ysc_testtrain_step(self, data):
        output = self.step(data, reward=1) # 这里是1还是0呢？
        # print(label, " buffer len is ",len(self.buffer[label]))
        return output

    def ysc_pre_train_pipeline(self, load=False):
        # 更新一个实时显示准确利率的功能
        # 调试stdp测试例子得时候,发现即使再最开始的时间步里,输出都会有20多次这种的输出
        # 我这边肯定 也得具有类似的效果吧
        
        global im   
        if load == False:
            df = pd.read_csv('output.csv', header=None) # 读取数据
            data = df.values.tolist()
            # orch.save(self.connection1.weight, 'weight_origin.pth') # 这个和self.connection1.parameters['weight'] 是一样的
            # self.save_state(filename = 'weight0.pth')
            print("开始训练")
            right_deque = deque(iterable=[0 for _ in range(100)], maxlen=100) # 用来统计最近100个的正确情况
            # right_predict_num = 0
            for index, row in enumerate(tqdm(data)):
                temp_input = torch.tensor(row, device=device).unsqueeze(0) # 增加了一个维度
                temp_predict = self.ysc_pretrain_step_and_predict(data=temp_input) # 返回预测结果
                real_label = self.new_check_label_from_data(temp_input)
                
                if index == 10:
                    self.save_state(filename = 'weight10.pth') # 这个是正确的， 和stdp算法内部的是一样的
                    
                if index == 2000:
                    self.save_state(filename = 'weight2000.pth') # 这个是正确的， 和stdp算法内部的是一样的
                
                # for temp_i in range(len(self.buffer)): 
                writer.add_scalars("buffer_len",{"len_0": len(self.buffer[0]),"len_1": len(self.buffer[1]),"len_2": len(self.buffer[2]),"len_3": len(self.buffer[3]) }, global_step=index) # 观察各个buffer 的情况

                if index > 100:
                    print(" assign_label = ", self.assign_label)
                    print(" deque", sum(right_deque))
                    print(" buffer_num", len(self.buffer[0]), len(self.buffer[1]), len(self.buffer[2]), len(self.buffer[3]))
                    
                if temp_predict == real_label:
                    right_deque.append(1)
                else:
                    right_deque.append(0)
                writer.add_scalar(tag="acc_predict", scalar_value= sum(right_deque) / len(right_deque), global_step=index) # 每次打印准确率

                im = self.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                     reshape=True, n_sqrt=int(np.sqrt(label_num)), side=16, im=im, wmax=1) #把权重 画出来 100 * 784 = 100 * 28 * 28
                

            self.ysc_pre_train_over_save() # 预训练结束， 开始 统计结果，然后进行测试
        else:
            print("加载数据，跳过训练过程")
            self.state_from_dict(filename=model_path, device=torch.device("cuda"))
            self.buffer = torch.load(buffer_path)
        

        self.assign_label_update() # 对结果进行统计，并保存到self.assign_label中

    def ysc_pretrain_step_and_predict(self, data):
        
        output = self.ysc_pretrain_step(data=data)          # 预训练 时间步
        # print(output)
        temp_cnt = [0 for _ in range(len(self.buffer))]     # 四个0
        temp_num = [0 for _ in range(len(self.buffer))]
        self.assign_label_update()                          # 统计一下每个神经元的归属label
        
        if self.assign_label == None: # 最开始跳过                  
            return 0

        for i in range(len(self.assign_label)):
            # print(i)
            temp_cnt[self.assign_label[i]] += output[0, i]  # 第一个维度是batch, 
            temp_num[self.assign_label[i]] += 1
        
        # print(temp_cnt, temp_num)

        predict_label = torch.argmax(torch.tensor(temp_cnt) / torch.tensor(temp_num)) # 这里和后面的唯一的区别就是,这里比较的是总和, 而下面比较的是平均值, 平均值 会更看重 突触的脉冲发放
        # tensor 保证除法可以 分别相除
        # 这里其实应该 比较的是 去掉0 之后的平均值, 或者是 去掉 一个阈值以下的 值得平均值, 但要是所有得都是1 那就 没有必要了
                    
        return predict_label # 返回预测label

    # 开始测试
    def ysc_test_pretrain(self):

        df2 = pd.read_csv('testdata.csv', header=None) # 读取数据
        data2 = df2.values.tolist()     

        print("开始测试")
        right_predict_num = 0
        
        f = True
        with torch.no_grad():
            # print(self.assign_label) # 全都是0 ？？？
            for index, row in enumerate(tqdm(data2)):
                real_label = self.new_check_label_from_data([row])
                data = torch.tensor(row, device=device).unsqueeze(0)
                temp_output = self.ysc_testtrain_step(data)
                print(temp_output)
            
                spike_output_test = [[]] * output_node_num # 再按照 train的方法统计一遍
                for o in range(self.assign_label.shape[0]): # 100
                    if spike_output_test[self.assign_label[o]] == []:
                        spike_output_test[self.assign_label[o]] = [temp_output[:, o]] # 每次把o位置的输出放在 这个位置label 对应的[]中，最后看哪个label对应的多，就是预测的结果是哪个
                    else:                       
                        spike_output_test[self.assign_label[o]].append(temp_output[:, o]) # 这里如果再引入奖励的话，再根据结果的是否正确，利用这次观测的数据，对网络进行一次调节，也可以对assign_label 的统计buffer 进行一次更新

                test_output = []
                for o in range(len(spike_output_test)): # 实际中可能再去考虑一下，emotion 变化的稳定性的问题 基本就完美了
                    if spike_output_test[o] == []: # 
                        pass
                        print("this is a zero")
                        test_output.append(torch.tensor(0, device=device).unsqueeze(0))
                    else:
                        test_output.append([sum(spike_output_test[o]) ]) # 这里使用总值试一下 #/ len(spike_output_test[o])]) # 发放的平均值， 为什么不是发放的总值呢
                # print(spike_output_test)
                # print(test_output)
                predict_label = torch.argmax(torch.tensor(test_output)) # 

                # print(predict_label, real_label)
                if f:                       
                    print(predict_label, real_label)
                    f = False
                if real_label == predict_label:
                    right_predict_num += 1
                
            print(right_predict_num) # 打印成功的个数
        

    def assign_label_update(self, newoutput=None, newlabel=None, weight=0):
        # 如果没有新的数据输入，则就是对 assign_label 进行一次计算，否则 会根据权重插入新数据，进而计算
        if newoutput != None:
            self.buffer[newlabel].append(newoutput)
        try:
            avg_buffer = [sum(self.buffer[i][-400:]) / len(self.buffer[i][-400:]) for i in range(len(self.buffer))] # sum_buffer 是一个求和之后 取平均的tensor  n * 1 * 100
            # 这里不如改成 200 试一下
            # 这里可以只使用 后面的数据进行统计  比如[-300:]
            # avg_buffer = [sum_buffer[i] / len(agent.buffer[i]) for i in range(len(agent.buffer))]
            assign_label = torch.argmax(torch.cat(avg_buffer, 0), 0) # n 个1*100 的list在第0个维度合并 -> n*100的tensor, 进而在第0个维度比哪个更大, 返回一个1维的tensor， 内容是index，[0,n)， 目前是0123
            # 这里的 100 个 0 、1、2、3 也就代表了， 当前那个神经元 可以代表的 类别是什么
            self.assign_label = assign_label # 初始化结束s
        except ZeroDivisionError:
            # 如果分母是零 说明是刚开始数据还不够的时候，就需要不管就行
            return 
            


if __name__ == "__main__":
    
    # 如果需要重新构造数据集的话，需要重新打开这个函数， 把其余部分注释掉

    # ysc_create_data_before_pretrain()

    ysc_robot_net = YscNet()
    # print(ysc_robot_net.connection1.weight)
    

    ysc_robot_net.ysc_pre_train_pipeline(load=False)

    # ysc_robot_net.ysc_test_pretrain()

    """ temp_input = torch.rand(1, 13).to(device=device)
    
    temp_output = ysc_robot_net.step(temp_input) """

    # print(temp_output.shape)

    
    
