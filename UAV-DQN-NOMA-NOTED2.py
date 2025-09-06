import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
from datetime import datetime
import imageio
import itertools

# 运行时调试日志：持续追加到 output_folder/runtime_log.txt
def append_runtime_log(output_folder, text):
    try:
        log_file = os.path.join(output_folder, 'runtime_log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
    except Exception:
        # 日志失败不影响训练
        pass

# 固定随机种子，确保每次运行代码时随机生成的用户位置一致
np.random.seed(1)  # 可以修改这个数字来获得不同的固定位置

# 训练参数配置 - 可以方便修改
TOTAL_EPISODES = 200  # 总训练轮次，可以修改这个数字
EXPERT_GUIDANCE_RATIO = 0.35  # 专家引导轮次比例（前35%）
TEST_EPISODES_RATIO = 0.05    # 测试轮次比例（后5%）

# 经验回放和训练参数配置
REPLAY_BUFFER_SIZE = 10000  # 经验回放区大小，可以修改这个数字
BATCH_SIZE = 256  # 批次大小，可以修改这个数字

# 计算探索阶段信息
T = 60  # 每个episode的时间步数
EXPERIENCES_PER_EPISODE = 3 * T + 1 * T  # 3个UAV + 1个BS，每个时间步产生4个经验
EXPLORATION_EPISODES = REPLAY_BUFFER_SIZE // EXPERIENCES_PER_EPISODE  # 探索阶段需要的episode数

# 自动计算各阶段轮次
EXPERT_EPISODES = int(TOTAL_EPISODES * EXPERT_GUIDANCE_RATIO)  # 专家引导轮次
TEST_EPISODES = int(TOTAL_EPISODES * TEST_EPISODES_RATIO)      # 测试轮次
TRAINING_EPISODES = TOTAL_EPISODES - TEST_EPISODES             # 训练轮次（包含专家引导）

# set up service area range
ServiceZone_X = 500
ServiceZone_Y = 500
Hight_limit_Z = 150

# set up users' speed
MAXUserspeed = 0.5 #m/s
UAV_Speed = 5 #m/s

# 系统配置 - 扩展为UAV-BS协作系统
UserNumberPerCell = 2 # user number per UAV
NumberOfUAVs = 3 # number of UAVs
NumberOfCells = NumberOfUAVs # Each UAV is responsible for one cell
NumberOfUAVUsers = NumberOfUAVs*UserNumberPerCell # UAV用户数 (6)
BSUserNumber = 6 # BS服务用户数
TotalUsers = NumberOfUAVUsers + BSUserNumber # 总用户数 (12)

# 频段配置
UAV_FREQ = 3.5 # UAV载波频率 (GHz)
BS_FREQ = 2.4  # BS载波频率 (GHz)
F_c = UAV_FREQ # 保持原有UAV频率

# 通信参数
Bandwidth = 30 #khz
R_require = 0.1 # QoS data rate requirement kb
Power_level= 3 # Since DQN can only solve discrete action spaces, we set several discrete power gears, Please note that the change of power leveal will require a reset on the action space

# 功率配置
amplification_constant = 10000 # Since the original power and noise values are sometims negligible, it may cause NAN data. We perform unified amplification to avoid data type errors
UAV_power_unit = 100 * amplification_constant # 100mW=20dBm
BS_power_unit = UAV_power_unit * 20 # BS功率单位是UAV的20倍
NoisePower = 10**(-9) * amplification_constant # noise power

# BS功率分配等级
BS_POWER_LEVELS = 4  # BS功率等级
UAV_POWER_LEVELS = 3  # UAV功率等级

# Define the neural network model for PyTorch
class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 40)
        self.fc2 = nn.Linear(40, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SystemModel(object):
    def __init__(
            self,
    ):
        # Initialize area
        self.Zone_border_X = ServiceZone_X
        self.Zone_border_Y = ServiceZone_Y
        self.Zone_border_Z = Hight_limit_Z

        # Initialize UAV and their location
        self.UAVspeed = UAV_Speed
        self.UAV_number = NumberOfUAVs
        self.UserperCell = UserNumberPerCell
        self.U_idx = np.arange(NumberOfUAVs) # set up serial number for UAVs
        self.PositionOfUAVs = pd.DataFrame(
           np.zeros((3,NumberOfUAVs)),
          columns=self.U_idx.tolist(),    # Data frame for saving UAVs' position
        )
        # 重新设计UAV初始位置，实现平面上均匀分布，高度统一为20m
        # 服务区域：500m x 500m，UAV高度：20m
        # UAV0: 左下区域 (125, 125, 20)
        # UAV1: 中心区域 (250, 250, 20)  
        # UAV2: 右上区域 (375, 375, 20)
        self.PositionOfUAVs.iloc[0, 0] = 125  # UAV0: x坐标
        self.PositionOfUAVs.iloc[1, 0] = 125  # UAV0: y坐标
        self.PositionOfUAVs.iloc[2, 0] = 20   # UAV0: z坐标
        self.PositionOfUAVs.iloc[0, 1] = 250  # UAV1: x坐标
        self.PositionOfUAVs.iloc[1, 1] = 250  # UAV1: y坐标
        self.PositionOfUAVs.iloc[2, 1] = 20   # UAV1: z坐标
        self.PositionOfUAVs.iloc[0, 2] = 375  # UAV2: x坐标
        self.PositionOfUAVs.iloc[1, 2] = 375  # UAV2: y坐标
        self.PositionOfUAVs.iloc[2, 2] = 20   # UAV2: z坐标

        # Initialize BS position (fixed)
        self.bs_position = np.array([250.0, 250.0, 30.0])  # 区域中心，高度30m
        
        # Initialize users and users' location (12个用户)
        self.User_number = TotalUsers
        self.K_idx = np.arange(TotalUsers) # set up serial number for users
        self.PositionOfUsers = pd.DataFrame(
           np.zeros((3,TotalUsers)),
          columns=self.K_idx.tolist(),    # Data frame for saving users' position
        )
        
        # 所有用户位置初始化 - 12个用户都使用随机位置
        self.PositionOfUsers.iloc[0,:] = np.random.uniform(0, ServiceZone_X, TotalUsers)
        self.PositionOfUsers.iloc[1,:] = np.random.uniform(0, ServiceZone_Y, TotalUsers)
        self.PositionOfUsers.iloc[2,:] = 0 # users' height is assumed to be 0

        # record initial state
        self.Init_PositionOfUsers = copy.deepcopy(self.PositionOfUsers)
        self.Init_PositionOfUAVs = copy.deepcopy(self.PositionOfUAVs)

        # initialize a array to store state
        # 状态空间：UAV位置(9) + UAV用户信道增益(6) = 15
        # UAV用户数量固定为6个，BS用户数量固定为6个
        self.State = np.zeros([1, NumberOfUAVs * 3 + NumberOfUAVUsers], dtype=float)

        # Create a data frame for storing transmit power
        self.Power_allocation_list = pd.DataFrame(
            np.ones((1, TotalUsers)),
            columns=np.arange(TotalUsers).tolist(),
        )
        self.Power_unit = UAV_power_unit
        # 初始化时所有用户使用UAV功率，后续会根据分配表动态调整
        # 注意：这里只是初始化，实际的功率分配会在用户分配确定后动态设置

        # data frame to save distance (UAV+BS to all users)
        self.Distence = pd.DataFrame(
            np.zeros((self.UAV_number + 1, self.User_number)),  # +1 for BS
            columns=np.arange(self.User_number).tolist(),)

        # data frame to save pathloss
        self.Propergation_Loss = pd.DataFrame(
            np.zeros((self.UAV_number + 1, self.User_number)),  # +1 for BS
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save channel gain
        self.ChannelGain_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save equivalent channel gain
        self.Eq_CG_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save SINR
        self.SINR_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save datarate
        self.Daterate = pd.DataFrame(
        np.zeros((1, self.User_number)),
        columns=np.arange(self.User_number).tolist(),)

        # amplification_constant as mentioned above
        self.amplification_constant = amplification_constant
        
        # Initialize position history for visualization
        self.uav_positions_history = []
        self.user_positions_history = []
        self.bs_positions_history = []
        
        # 添加详细数据记录功能
        self.debug_data = {
            'episode_data': [],
            'expert_guidance_data': [],
            'qos_violation_data': []
        }
        
        # 精简启动打印（不再向终端输出初始化位置）


    def User_randomMove(self,MAXspeed,NumberofUsers):
        self.PositionOfUsers.iloc[[0,1],:] += np.random.randn(2,NumberofUsers)*MAXspeed # users random move
        return


    def Get_Distance_U2K(self,UAV_Position,User_Position,UAVsnumber,Usersnumber): # this function is for calculating the distance between users and UAVs/BS

        # Calculate distances from UAVs to users
        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                self.Distence.iloc[i,j] = np.linalg.norm(UAV_Position.iloc[:,i]-User_Position.iloc[:,j]) # calculate Distance between UAV i and User j
        
        # Calculate distances from BS to users
        for j in range(Usersnumber):
            self.Distence.iloc[UAVsnumber,j] = np.linalg.norm(self.bs_position-User_Position.iloc[:,j]) # calculate Distance between BS and User j

        return self.Distence


    def Get_Propergation_Loss(self,distence_U2K,UAV_Position,UAVsnumber,Usersnumber,f_c): # this function is for calculating the pathloss between users and UAVs/BS

        # Calculate UAV pathloss (UAV用户使用UAV频段)
        for i in range(UAVsnumber):# Calculate average loss for each user,  this pathloss model is for 22.5m<h<300m d(2d)<4km
            for j in range(Usersnumber):
                UAV_Hight=UAV_Position.iloc[2,i]
                D_H = np.sqrt(np.square(distence_U2K.iloc[i,j])-np.square(UAV_Hight)) # calculate distance
                # calculate the possibility of LOS/NLOS
                d_0 = np.max([(294.05*math.log(UAV_Hight,10)-432.94),18])
                p_1 = 233.98*math.log(UAV_Hight,10) - 0.95
                if D_H <= d_0:
                    P_Los = 1.0
                else:
                    P_Los = d_0/D_H + math.exp(-(D_H/p_1)*(1-(d_0/D_H)))

                if P_Los>1:
                    P_Los = 1

                P_NLos = 1 - P_Los

                #calculate the passlos for LOS/NOLS
                L_Los = 30.9 + (22.25-0.5*math.log(UAV_Hight,10))*math.log(distence_U2K.iloc[i,j],10) + 20*math.log(f_c,10)
                L_NLos = np.max([L_Los,32.4+(43.2-7.6*math.log(UAV_Hight,10))*math.log(distence_U2K.iloc[i,j],10)+20*math.log(f_c,10)])

                Avg_Los = P_Los*L_Los + P_NLos*L_NLos # average pathloss
                gain = np.random.rayleigh(scale=1, size=None)*pow(10,(-Avg_Los/10)) # random fading
                self.Propergation_Loss.iloc[i,j] = gain #save pathloss

        # Calculate BS pathloss (BS用户使用BS频段)
        for j in range(Usersnumber):
            bs_distance = distence_U2K.iloc[UAVsnumber,j]
            # 城市宏小区路径损耗模型（考虑频率因子，基于2 GHz基线校正到 f_c）
            # PL(dB) = 128.1 + 37.6*log10(d_km) + 20*log10(f_c/2.0)
            # 其中 f_c 为 GHz
            freq_factor_db = 20 * math.log10(max(f_c / 2.0, 1e-6))
            PL = 128.1 + 37.6 * math.log10(bs_distance / 1000) + freq_factor_db
            gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-PL / 10))
            self.Propergation_Loss.iloc[UAVsnumber,j] = gain

        return self.Propergation_Loss


    def Get_Propergation_Loss_UAV(self, distence_U2K, UAV_Position, UAVsnumber, Usersnumber):
        """仅计算UAV频段的路径损耗：
        - 填充UAV行（0..UAVsnumber-1），使用UAV频段 F_c
        - 将BS行（index=UAVsnumber）置零，避免跨系统干扰
        """
        # 先清零
        self.Propergation_Loss.iloc[:, :] = 0.0

        f_c = UAV_FREQ  # 使用UAV频段
        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                UAV_Hight = UAV_Position.iloc[2, i]
                D_H = np.sqrt(np.square(distence_U2K.iloc[i, j]) - np.square(UAV_Hight))
                d_0 = np.max([(294.05 * math.log(UAV_Hight, 10) - 432.94), 18])
                p_1 = 233.98 * math.log(UAV_Hight, 10) - 0.95
                if D_H <= d_0:
                    P_Los = 1.0
                else:
                    P_Los = d_0 / D_H + math.exp(-(D_H / p_1) * (1 - (d_0 / D_H)))
                if P_Los > 1:
                    P_Los = 1
                P_NLos = 1 - P_Los
                L_Los = 30.9 + (22.25 - 0.5 * math.log(UAV_Hight, 10)) * math.log(distence_U2K.iloc[i, j], 10) + 20 * math.log(f_c, 10)
                L_NLos = np.max([L_Los, 32.4 + (43.2 - 7.6 * math.log(UAV_Hight, 10)) * math.log(distence_U2K.iloc[i, j], 10) + 20 * math.log(f_c, 10)])
                Avg_Los = P_Los * L_Los + P_NLos * L_NLos
                gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-Avg_Los / 10))
                self.Propergation_Loss.iloc[i, j] = gain

        # BS行清零（与UAV频段解耦）
        self.Propergation_Loss.iloc[UAVsnumber, :] = 0.0
        return self.Propergation_Loss


    def Get_Propergation_Loss_BS(self, distence_U2K, UAV_Position, UAVsnumber, Usersnumber, f_c):
        """仅计算BS频段的路径损耗：
        - 填充BS行（index=UAVsnumber），使用传入的 BS 频段
        - 将UAV行（0..UAVsnumber-1）置零
        """
        # 先清零
        self.Propergation_Loss.iloc[:, :] = 0.0

        # 仅计算BS行
        for j in range(Usersnumber):
            bs_distance = distence_U2K.iloc[UAVsnumber, j]
            freq_factor_db = 20 * math.log10(max(f_c / 2.0, 1e-6))
            PL = 128.1 + 37.6 * math.log10(bs_distance / 1000) + freq_factor_db
            gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-PL / 10))
            self.Propergation_Loss.iloc[UAVsnumber, j] = gain

        # UAV行清零（与BS频段解耦）
        for i in range(UAVsnumber):
            self.Propergation_Loss.iloc[i, :] = 0.0
        return self.Propergation_Loss

    def Get_Channel_Gain_NOMA(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power): # this function is for calculating channel gain

        for j in range(Usersnumber):  # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service
            i_Server_UAV = UserAssociationlist[j]  # UserAssociationlist现在是numpy数组
            Signal_power = self.amplification_constant * PropergationLosslist.iloc[i_Server_UAV, j]
            ChannelGain = Signal_power / ( Noise_Power) # calculate channel gain
            self.ChannelGain_list.iloc[0, j] = ChannelGain # save channel gain

        return self.ChannelGain_list


    def Get_Eq_CG(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power): #This function is used to calculate the equivalent channel gain to determine SIC decoding order

        for j in range(Usersnumber):  # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist[j]  # UserAssociationlist现在是numpy数组
            Signal_power = 100 * self.amplification_constant * PropergationLosslist.iloc[i_Server_UAV, j] # Assuming unit power to calculate equivalent channel gain
            I_inter_cluster = 0

            for j_idx in range(Usersnumber):  # calculate Interference for user j
                if UserAssociationlist[j_idx] == i_Server_UAV: # if the user j_idx is user j, pass
                    pass
                else:
                    Inter_UAV = UserAssociationlist[j_idx]  # find the inter UAV connected with user j_idx
                    # 频段隔离：仅累加同系统（同频）干扰
                    if (i_Server_UAV < UAVsnumber and Inter_UAV < UAVsnumber) or (i_Server_UAV == UAVsnumber and Inter_UAV == UAVsnumber):
                        I_inter_cluster = I_inter_cluster + (
                                100 * self.amplification_constant * PropergationLosslist.iloc[Inter_UAV, j]) # calculte and add inter cluster interference

            Eq_CG = Signal_power / (I_inter_cluster + Noise_Power) # calculate equivalent channel gain for user j
            self.Eq_CG_list.iloc[0, j] = Eq_CG

        return self.Eq_CG_list


    def Get_SINR_NNOMA(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,ChannelGain_list,Noise_Power):
        #This function is to calculate the SINR for every users

        for j in range(Usersnumber): # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist[j]  # UserAssociationlist现在是numpy数组
            Signal_power = self.Power_allocation_list.iloc[0,j] * PropergationLosslist.iloc[i_Server_UAV,j] # read the sinal power from power allocation list
            I_inter_cluster = 0

            for j_idx in range(Usersnumber): # calculate Interference for user j
                if UserAssociationlist[j_idx] == i_Server_UAV:
                    if ChannelGain_list.iloc[0,j] < ChannelGain_list.iloc[0,j_idx] and j!=j_idx: #find 'stronger' users in same cluster to count intra cluster interference
                        I_inter_cluster = I_inter_cluster + (
                                    self.Power_allocation_list.iloc[0, j_idx] * PropergationLosslist.iloc[
                                i_Server_UAV, j])  #calculate intra cluster interference

                else:
                    Inter_UAV = UserAssociationlist[j_idx] # calculate inter cluster interference from other UAVs
                    # 频段隔离：仅累加同系统（同频）干扰
                    if (i_Server_UAV < UAVsnumber and Inter_UAV < UAVsnumber) or (i_Server_UAV == UAVsnumber and Inter_UAV == UAVsnumber):
                        I_inter_cluster = I_inter_cluster + (self.Power_allocation_list.iloc[0,j_idx] * PropergationLosslist.iloc[Inter_UAV,j])#

            SINR = Signal_power/(I_inter_cluster + Noise_Power) # calculate SINR and save it
            self.SINR_list.iloc[0,j] = SINR

        return self.SINR_list


    def Calcullate_Datarate(self,SINRlist,Usersnumber,B): # calculate data rate for all users

        for j in range(Usersnumber):

            if SINRlist.iloc[0,j] <=0:
                # suppress noisy terminal output; detailed QoS/SINR logs are recorded elsewhere
                # 即使SINR <= 0，也设置一个很小的正数值，避免数据率为0
                self.Daterate.iloc[0,j] = B*math.log(1+1e-10,2)  # 设置一个很小的正数
            else:
                self.Daterate.iloc[0,j] = B*math.log((1+SINRlist.iloc[0,j]),2)

        SumDataRate = sum(self.Daterate.iloc[0,:])
        Worst_user_rate = min(self.Daterate.iloc[0,:])
        return self.Daterate,SumDataRate,Worst_user_rate


    def Reset_position(self): # save initial state for environment reset
        self.PositionOfUsers = copy.deepcopy(self.Init_PositionOfUsers)
        self.PositionOfUAVs = copy.deepcopy(self.Init_PositionOfUAVs)
        # Clear position history for new episode
        self.uav_positions_history = []
        self.user_positions_history = []
        return
        
    def record_positions(self):
        """Record current positions for visualization"""
        # 添加调试信息：打印当前UAV位置
        # 精简记录位置时的打印（不向终端输出）
        
        self.uav_positions_history.append(copy.deepcopy(self.PositionOfUAVs.values))
        self.user_positions_history.append(copy.deepcopy(self.PositionOfUsers.values))
        self.bs_positions_history.append(copy.deepcopy(self.bs_position.reshape(3, 1)))
    
    def record_episode_data(self, episode, step, user_association, data_rate, sum_rate, expert_prob, epsilon):
        """记录每个episode的详细数据"""
        # 确保数据类型可以被JSON序列化
        uav_positions = self.PositionOfUAVs.values.tolist() if hasattr(self.PositionOfUAVs.values, 'tolist') else self.PositionOfUAVs.values.tolist()
        user_positions = self.PositionOfUsers.values.tolist() if hasattr(self.PositionOfUsers.values, 'tolist') else self.PositionOfUsers.values.tolist()
        bs_position = self.bs_position.tolist() if hasattr(self.bs_position, 'tolist') else self.bs_position.tolist()
        user_association_list = user_association.tolist() if hasattr(user_association, 'tolist') else list(user_association)
        data_rates_list = data_rate.values[0].tolist() if hasattr(data_rate.values[0], 'tolist') else data_rate.values[0].tolist()
        
        episode_data = {
            'episode': int(episode),
            'step': int(step),
            'uav_positions': uav_positions,
            'user_positions': user_positions,
            'bs_position': bs_position,
            'user_association': user_association_list,
            'data_rates': data_rates_list,
            'sum_rate': float(sum_rate),
            'expert_prob': float(expert_prob),
            'epsilon': float(epsilon),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.debug_data['episode_data'].append(episode_data)
    
    def record_expert_guidance_data(self, episode, step, uav_id, uav_position, served_users, 
                                   user_positions, midpoint, expert_action, distance_to_midpoint):
        """记录专家引导的详细数据"""
        # 确保数据类型可以被JSON序列化
        served_users_list = served_users.tolist() if hasattr(served_users, 'tolist') else list(served_users)
        
        expert_data = {
            'episode': int(episode),
            'step': int(step),
            'uav_id': int(uav_id),
            'uav_position': uav_position.tolist(),
            'served_users': served_users_list,
            'user_positions': [pos.tolist() for pos in user_positions],
            'midpoint': midpoint.tolist(),
            'expert_action': int(expert_action),
            'distance_to_midpoint': float(distance_to_midpoint),
            'movement_action': int(expert_action % 7),
            'power_action': int(expert_action // 7),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.debug_data['expert_guidance_data'].append(expert_data)
    
    def record_qos_violation_data(self, episode, step, uav_id, served_users, data_rates, qos_violation_count, 
                                 violation_ratio, avg_qos_satisfaction, penalty_factor, reward):
        """记录QoS违反的详细数据"""
        # 确保数据类型可以被JSON序列化
        served_users_list = served_users.tolist() if hasattr(served_users, 'tolist') else list(served_users)
        
        qos_data = {
            'episode': episode,
            'step': step,
            'uav_id': uav_id,
            'served_users': served_users_list,
            'user_data_rates': [float(data_rates[user_id]) for user_id in served_users],
            'qos_requirement': float(R_require),
            'qos_violation_count': int(qos_violation_count),
            'violation_ratio': float(violation_ratio),
            'avg_qos_satisfaction': float(avg_qos_satisfaction),
            'penalty_factor': float(penalty_factor),
            'reward': float(reward),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.debug_data['qos_violation_data'].append(qos_data)
    
    def save_debug_data(self, output_folder):
        """保存调试数据到文件"""
        import json
        
        # 保存episode数据
        episode_file = os.path.join(output_folder, 'episode_debug_data.json')
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(self.debug_data['episode_data'], f, indent=2, ensure_ascii=False)
        
        # 保存专家引导数据
        expert_file = os.path.join(output_folder, 'expert_guidance_debug_data.json')
        with open(expert_file, 'w', encoding='utf-8') as f:
            json.dump(self.debug_data['expert_guidance_data'], f, indent=2, ensure_ascii=False)
        
        # 保存QoS违反数据
        qos_file = os.path.join(output_folder, 'qos_violation_debug_data.json')
        with open(qos_file, 'w', encoding='utf-8') as f:
            json.dump(self.debug_data['qos_violation_data'], f, indent=2, ensure_ascii=False)
        
        # 生成统计报告
        self.generate_debug_report(output_folder)
        
        print(f"调试数据已保存到: {output_folder}")
    
    def generate_debug_report(self, output_folder):
         """生成调试报告"""
         report_file = os.path.join(output_folder, 'debug_report.txt')
         
         with open(report_file, 'w', encoding='utf-8') as f:
             f.write("=== UAV-DQN-NOMA 系统调试报告 ===\n\n")
             
             # 专家引导统计
             expert_data = self.debug_data['expert_guidance_data']
             if expert_data:
                 f.write("1. 专家引导统计:\n")
                 f.write(f"   总专家引导次数: {len(expert_data)}\n")
                 
                 # 按UAV统计
                 for uav_id in range(NumberOfUAVs):
                     uav_expert_data = [d for d in expert_data if d['uav_id'] == uav_id]
                     f.write(f"   UAV{uav_id} 专家引导次数: {len(uav_expert_data)}\n")
                     
                     if uav_expert_data:
                         # 分析UAV1的移动情况
                         if uav_id == 1:
                             f.write(f"   UAV1 详细分析:\n")
                             keep_position_count = sum(1 for d in uav_expert_data if d['movement_action'] == 6)
                             f.write(f"     保持位置次数: {keep_position_count}/{len(uav_expert_data)}\n")
                             
                             # 分析距离分布
                             distances = [d['distance_to_midpoint'] for d in uav_expert_data]
                             f.write(f"     到中点距离统计: 平均={np.mean(distances):.2f}m, 最小={np.min(distances):.2f}m, 最大={np.max(distances):.2f}m\n")
                             
                             # 分析最近几次的专家引导
                             recent_data = uav_expert_data[-5:] if len(uav_expert_data) >= 5 else uav_expert_data
                             f.write(f"     最近{len(recent_data)}次专家引导:\n")
                             for i, data in enumerate(recent_data):
                                 f.write(f"       {i+1}. 距离={data['distance_to_midpoint']:.2f}m, 动作={data['movement_action']}, 功率={data['power_action']}\n")
                         
                         # 新增：分析UAV2的移动情况（重点关注）
                         elif uav_id == 2:
                             f.write(f"   UAV2 详细分析:\n")
                             keep_position_count = sum(1 for d in uav_expert_data if d['movement_action'] == 6)
                             f.write(f"     保持位置次数: {keep_position_count}/{len(uav_expert_data)}\n")
                             
                             # 分析距离分布
                             distances = [d['distance_to_midpoint'] for d in uav_expert_data]
                             f.write(f"     到中点距离统计: 平均={np.mean(distances):.2f}m, 最小={np.min(distances):.2f}m, 最大={np.max(distances):.2f}m\n")
                             
                             # 分析激进移动策略的使用情况
                             aggressive_moves = sum(1 for d in uav_expert_data if d['distance_to_midpoint'] > 100)
                             f.write(f"     激进移动策略使用次数: {aggressive_moves}/{len(uav_expert_data)}\n")
                             
                             # 分析最近几次的专家引导
                             recent_data = uav_expert_data[-5:] if len(uav_expert_data) >= 5 else uav_expert_data
                             f.write(f"     最近{len(recent_data)}次专家引导:\n")
                             for i, data in enumerate(recent_data):
                                 f.write(f"       {i+1}. 距离={data['distance_to_midpoint']:.2f}m, 动作={data['movement_action']}, 功率={data['power_action']}\n")
             
             # QoS违反统计
             qos_data = self.debug_data['qos_violation_data']
             if qos_data:
                 f.write("\n2. QoS违反统计:\n")
                 f.write(f"   总QoS检查次数: {len(qos_data)}\n")
                 
                 total_violations = sum(d['qos_violation_count'] for d in qos_data)
                 f.write(f"   总QoS违反次数: {total_violations}\n")
                 
                 avg_violation_ratio = np.mean([d['violation_ratio'] for d in qos_data])
                 f.write(f"   平均违反比例: {avg_violation_ratio:.3f}\n")
                 
                 # 按UAV统计QoS违反
                 for uav_id in range(NumberOfUAVs):
                     uav_qos_data = [d for d in qos_data if d['uav_id'] == uav_id]
                     if uav_qos_data:
                         uav_violations = sum(d['qos_violation_count'] for d in uav_qos_data)
                         f.write(f"   UAV{uav_id} QoS违反次数: {uav_violations}\n")
             
             f.write("\n=== 报告生成完成 ===\n")
     
    def generate_simulation_parameters_report(self, output_folder, total_episodes, expert_episodes, test_episodes, 
                                           training_episodes, uav_throughput_seq, bs_throughput_seq, 
                                           system_throughput_seq, uav_worstuser_tp_seq, bs_worstuser_tp_seq, 
                                           system_worstuser_tp_seq, datarate_seq, uav_reward_seq, 
                                           bs_reward_seq, uav_final_datarate_seq, bs_final_datarate_seq):
         """生成仿真参数报告"""
         report_file = os.path.join(output_folder, 'simulation_parameters_report.txt')
         
         with open(report_file, 'w', encoding='utf-8') as f:
             f.write("=== UAV-DQN-NOMA 系统仿真参数报告 ===\n\n")
             
             # 系统配置参数
             f.write("1. 系统配置参数:\n")
             f.write(f"   服务区域: {ServiceZone_X}m × {ServiceZone_Y}m × {Hight_limit_Z}m\n")
             f.write(f"   UAV数量: {NumberOfUAVs}\n")
             f.write(f"   每UAV服务用户数: {UserNumberPerCell}\n")
             f.write(f"   总用户数: {TotalUsers}\n")
             f.write(f"   UAV用户数: {NumberOfUAVUsers}\n")
             f.write(f"   BS用户数: {BSUserNumber}\n")
             f.write(f"   UAV载波频率: {UAV_FREQ} GHz\n")
             f.write(f"   BS载波频率: {BS_FREQ} GHz\n")
             f.write(f"   带宽: {Bandwidth} kHz\n")
             f.write(f"   QoS要求: {R_require} kb\n")
             f.write(f"   UAV功率等级数: {UAV_POWER_LEVELS}\n")
             f.write(f"   BS功率等级数: {BS_POWER_LEVELS}\n")
             f.write(f"   UAV功率单位: {UAV_power_unit/amplification_constant:.0f} mW\n")
             f.write(f"   BS功率单位: {BS_power_unit/amplification_constant:.0f} mW\n")
             f.write(f"   噪声功率: {NoisePower/amplification_constant:.2e} W\n")
             f.write(f"   用户最大速度: {MAXUserspeed} m/s\n")
             f.write(f"   UAV速度: {UAV_Speed} m/s\n\n")
             
             # 训练参数
             f.write("2. 训练参数:\n")
             f.write(f"   总训练轮次: {total_episodes}\n")
             f.write(f"   专家引导轮次: {expert_episodes}\n")
             f.write(f"   纯训练轮次: {training_episodes - expert_episodes}\n")
             f.write(f"   测试轮次: {test_episodes}\n")
             f.write(f"   每轮时间步数: 60\n")
             f.write(f"   用户分配周期: 400步\n")
             f.write(f"   初始ε值: 0.9\n")
             f.write(f"   最小ε值: 0.05\n")
             f.write(f"   学习率: 0.001\n")
             f.write(f"   批次大小: 128\n")
             f.write(f"   目标网络更新频率: 600步\n")
             f.write(f"   经验回放缓冲区大小: 10000\n\n")
             
             # 性能统计
             f.write("3. 性能统计:\n")
             if len(uav_throughput_seq) > 0:
                 f.write(f"   UAV系统平均吞吐量: {np.mean(uav_throughput_seq):.2f}\n")
                 f.write(f"   UAV系统最大吞吐量: {np.max(uav_throughput_seq):.2f}\n")
                 f.write(f"   UAV系统最小吞吐量: {np.min(uav_throughput_seq):.2f}\n")
                 f.write(f"   UAV系统吞吐量标准差: {np.std(uav_throughput_seq):.2f}\n")
             
             if len(bs_throughput_seq) > 0:
                 f.write(f"   BS系统平均吞吐量: {np.mean(bs_throughput_seq):.2f}\n")
                 f.write(f"   BS系统最大吞吐量: {np.max(bs_throughput_seq):.2f}\n")
                 f.write(f"   BS系统最小吞吐量: {np.min(bs_throughput_seq):.2f}\n")
                 f.write(f"   BS系统吞吐量标准差: {np.std(bs_throughput_seq):.2f}\n")
             
             if len(system_throughput_seq) > 0:
                 f.write(f"   系统总平均吞吐量: {np.mean(system_throughput_seq):.2f}\n")
                 f.write(f"   系统总最大吞吐量: {np.max(system_throughput_seq):.2f}\n")
                 f.write(f"   系统总最小吞吐量: {np.min(system_throughput_seq):.2f}\n")
                 f.write(f"   系统总吞吐量标准差: {np.std(system_throughput_seq):.2f}\n")
             
             if len(uav_worstuser_tp_seq) > 0:
                 f.write(f"   UAV最差用户平均吞吐量: {np.mean(uav_worstuser_tp_seq):.2f}\n")
                 f.write(f"   UAV最差用户最大吞吐量: {np.max(uav_worstuser_tp_seq):.2f}\n")
                 f.write(f"   UAV最差用户最小吞吐量: {np.min(uav_worstuser_tp_seq):.2f}\n")
             
             if len(bs_worstuser_tp_seq) > 0:
                 f.write(f"   BS最差用户平均吞吐量: {np.mean(bs_worstuser_tp_seq):.2f}\n")
                 f.write(f"   BS最差用户最大吞吐量: {np.max(bs_worstuser_tp_seq):.2f}\n")
                 f.write(f"   BS最差用户最小吞吐量: {np.min(bs_worstuser_tp_seq):.2f}\n")
             
             if len(system_worstuser_tp_seq) > 0:
                 f.write(f"   系统最差用户平均吞吐量: {np.mean(system_worstuser_tp_seq):.2f}\n")
                 f.write(f"   系统最差用户最大吞吐量: {np.max(system_worstuser_tp_seq):.2f}\n")
                 f.write(f"   系统最差用户最小吞吐量: {np.min(system_worstuser_tp_seq):.2f}\n")
             
             if len(datarate_seq) > 0:
                 f.write(f"   最后一个episode平均数据率: {np.mean(datarate_seq):.2f}\n")
                 f.write(f"   最后一个episode最大数据率: {np.max(datarate_seq):.2f}\n")
                 f.write(f"   最后一个episode最小数据率: {np.min(datarate_seq):.2f}\n")
             
             # 新增：reward和datarate统计
             f.write("\n4. Reward和DataRate统计:\n")
             f.write(f"   UAV平均每轮次reward: {np.mean(uav_reward_seq):.2f}\n")
             f.write(f"   UAV最大每轮次reward: {np.max(uav_reward_seq):.2f}\n")
             f.write(f"   UAV最小每轮次reward: {np.min(uav_reward_seq):.2f}\n")
             f.write(f"   BS平均每轮次reward: {np.mean(bs_reward_seq):.2f}\n")
             f.write(f"   BS最大每轮次reward: {np.max(bs_reward_seq):.2f}\n")
             f.write(f"   BS最小每轮次reward: {np.min(bs_reward_seq):.2f}\n")
             
             f.write(f"   UAV平均最后一步datarate: {np.mean(uav_final_datarate_seq):.2f}\n")
             f.write(f"   UAV最大最后一步datarate: {np.max(uav_final_datarate_seq):.2f}\n")
             f.write(f"   UAV最小最后一步datarate: {np.min(uav_final_datarate_seq):.2f}\n")
             f.write(f"   BS平均最后一步datarate: {np.mean(bs_final_datarate_seq):.2f}\n")
             f.write(f"   BS最大最后一步datarate: {np.max(bs_final_datarate_seq):.2f}\n")
             f.write(f"   BS最小最后一步datarate: {np.min(bs_final_datarate_seq):.2f}\n")
             
             f.write("\n=== 仿真参数报告生成完成 ===\n")

    def bs_uav_dynamic_user_assignment(self):
        """动态用户分配算法 - 与参考代码完全一致"""
        user_xy = self.PositionOfUsers.iloc[:2, :].T.values  # shape: (12, 2)
        bs_xy = self.bs_position[:2]
        uav_xy = self.PositionOfUAVs.iloc[:2, :].T.values  # shape: (3, 2)
        
        lambda_balance = 0.5  # UAV最大距离权重
        mu_bs = 5.0           # BS用户距离权重
        # 新增：UAV位置平衡性权重
        uav_position_balance_weight = 0.3  # UAV位置平衡性权重
        min_obj = float('inf')
        best_assign = None
        best_uav_assign = None
        user_idx = list(range(TotalUsers))
        
        # 枚举所有6人组合作为UAV用户
        for uav_users in itertools.combinations(user_idx, 6):
            uav_users = list(uav_users)
            
            # 这6人分成3对
            for pairs in itertools.combinations(itertools.combinations(uav_users, 2), 3):
                flat = [i for pair in pairs for i in pair]
                if sorted(flat) != sorted(uav_users):
                    continue
                
                # 剩下6人为BS用户
                bs_users = list(set(user_idx) - set(uav_users))
                
                # UAV三对距离检查
                uav_dists = [np.linalg.norm(user_xy[a] - user_xy[b]) for a, b in pairs]
                if any(d > 200 for d in uav_dists):
                    continue
                
                uav_sum = sum(uav_dists)
                uav_max = max(uav_dists)
                
                # UAV三对与三架UAV的距离矩阵
                pair_centers = [0.5 * (user_xy[a] + user_xy[b]) for a, b in pairs]
                dist_mat = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        dist_mat[i, j] = np.linalg.norm(pair_centers[i] - uav_xy[j])
                
                # 枚举三对分配给三架UAV的所有方式
                min_assign_sum = float('inf')
                best_assign_idx = None
                for perm in itertools.permutations([0, 1, 2]):
                    assign_sum = sum(dist_mat[i, perm[i]] for i in range(3))
                    if assign_sum < min_assign_sum:
                        min_assign_sum = assign_sum
                        best_assign_idx = perm
                
                # BS用户到BS距离和
                bs_sum = sum(np.linalg.norm(user_xy[i] - bs_xy) for i in bs_users)
                
                # 计算UAV位置平衡性指标
                # 计算每个UAV到其分配用户对中心的距离方差，方差越小表示分布越平衡
                uav_distances = []
                for i in range(3):
                    uav_pos = uav_xy[i]
                    pair_center = pair_centers[best_assign_idx[i]]
                    uav_distances.append(np.linalg.norm(uav_pos - pair_center))
                
                # 计算距离方差（平衡性指标）
                uav_distance_variance = np.var(uav_distances) if len(uav_distances) > 1 else 0
                
                # 改进的目标函数：增加UAV位置平衡性约束
                obj = uav_sum + lambda_balance * uav_max + mu_bs * bs_sum + min_assign_sum + uav_position_balance_weight * uav_distance_variance
                if obj < min_obj:
                    min_obj = obj
                    best_assign = (pairs, bs_users)
                    best_uav_assign = best_assign_idx
        
        # 生成分配结果
        user_association = np.zeros(TotalUsers, dtype=int)
        
        # 检查是否找到有效分配
        if best_assign is None or best_uav_assign is None:
            print("警告: 未找到完全满足200m约束的UAV用户分配，选择超距最小的分配")
            
            # 备用策略：选择超距最小的那组三对
            min_violation = float('inf')
            best_backup_assign = None
            best_backup_uav_assign = None
            
            # 重新遍历所有可能的6人组合
            for uav_users in itertools.combinations(user_idx, 6):
                uav_users = list(uav_users)
                
                # 这6人分成3对
                for pairs in itertools.combinations(itertools.combinations(uav_users, 2), 3):
                    flat = [i for pair in pairs for i in pair]
                    if sorted(flat) != sorted(uav_users):
                        continue
                    
                    # 剩下6人为BS用户
                    bs_users = list(set(user_idx) - set(uav_users))
                    
                    # 计算每对的距离和最大超距
                    uav_dists = [np.linalg.norm(user_xy[a] - user_xy[b]) for a, b in pairs]
                    max_violation = max(0, max(uav_dists) - 200)  # 超出200m的部分
                    
                    if max_violation < min_violation:
                        min_violation = max_violation
                        best_backup_assign = (pairs, bs_users)
                        
                        # 计算最佳UAV分配
                        pair_centers = [0.5 * (user_xy[a] + user_xy[b]) for a, b in pairs]
                        dist_mat = np.zeros((3, 3))
                        for i in range(3):
                            for j in range(3):
                                dist_mat[i, j] = np.linalg.norm(pair_centers[i] - uav_xy[j])
                        
                        min_assign_sum = float('inf')
                        best_assign_idx = None
                        for perm in itertools.permutations([0, 1, 2]):
                            assign_sum = sum(dist_mat[i, perm[i]] for i in range(3))
                            if assign_sum < min_assign_sum:
                                min_assign_sum = assign_sum
                                best_assign_idx = perm
                        
                        best_backup_uav_assign = best_assign_idx
            
            # 使用备用分配
            best_assign = best_backup_assign
            best_uav_assign = best_backup_uav_assign
            print(f"选择超距最小的分配，最大超距: {min_violation:.2f}m")
        
        # UAV用户分配
        pairs, bs_users = best_assign
        for i, (a, b) in enumerate(pairs):
            uav_id = best_uav_assign[i]
            user_association[a] = uav_id
            user_association[b] = uav_id
        
        # BS用户分配
        for idx in bs_users:
            user_association[idx] = NumberOfUAVs  # BS的ID是3
        
        # 验证分配结果
        # 精简用户分配打印（不向终端输出）
        # 打印分配结果（只在正式训练阶段）
        # 注意：这里无法直接判断是否在探索阶段，所以注释掉打印
        # for uav_id in range(NumberOfUAVs):
        #     uav_users = np.where(user_association == uav_id)[0]
        #     print(f"  UAV{uav_id}: {uav_users}")
        # bs_users = np.where(user_association == NumberOfUAVs)[0]
        # print(f"  BS: {bs_users}")
        
        # 根据分配结果动态设置功率
        self.update_power_allocation_based_on_assignment(user_association)
        
        return user_association
    
    def update_power_allocation_based_on_assignment(self, user_association):
        """根据用户分配表动态更新功率分配"""
        for user_id in range(TotalUsers):
            if user_association[user_id] < NumberOfUAVs:  # UAV用户
                self.Power_allocation_list.iloc[0, user_id] = UAV_power_unit
            else:  # BS用户
                self.Power_allocation_list.iloc[0, user_id] = BS_power_unit

    def get_uav_power_levels(self, user_association_list):
        uav_power_levels = []
        for uav in range(NumberOfUAVs):
            user_idx = np.where(user_association_list == uav)[0]
            # 获取分配给该UAV的用户（不再限制为前6个）
            if len(user_idx) >= 2:
                p1 = self.Power_allocation_list.iloc[0, user_idx[0]]
                p2 = self.Power_allocation_list.iloc[0, user_idx[1]]
                p1_level = round(p1 / self.Power_unit)
                p2_level = round(p2 / self.Power_unit)
                uav_power_levels.append((p1_level, p2_level))
            else:
                uav_power_levels.append((0, 0))
        return uav_power_levels

    def Create_state_Noposition(self,serving_UAV,User_association_list,User_Channel_Gain):
        # Create state, pay attention we need to ensure UAVs and users who are making decisions always input at the fixed neural node to achieve MDQN
        UAV_position_copy = copy.deepcopy(self.PositionOfUAVs.values)
        UAV_position_copy[:,[0,serving_UAV]] = UAV_position_copy[:,[serving_UAV,0]] # adjust the input node of serving UAV to ensure it is fixed
        User_Channel_Gain_copy = copy.deepcopy(User_Channel_Gain.values[0])

        for UAV in range(NumberOfUAVs):
            self.State[0, 3 * UAV:3 * UAV + 3] = UAV_position_copy[:, UAV].T # save UAV positions as a part of the state

        User_association_copy = copy.deepcopy(User_association_list)
        desirable_user = np.where(User_association_copy==serving_UAV)[0] # find out the current served users

        for i in range(len(desirable_user)):
             User_Channel_Gain_copy[i],User_Channel_Gain_copy[desirable_user[i]] = User_Channel_Gain_copy[desirable_user[i]],User_Channel_Gain_copy[i] # Similarly, adjust the input node of the current served users

        # 保存UAV用户（根据用户分配表确定）的信道增益到状态中
        # UAV用户数量固定为6个，需要根据分配表找出这6个用户
        uav_users = []
        for user_id in range(TotalUsers):
            if User_association_copy[user_id] < NumberOfUAVs:  # 是UAV用户
                uav_users.append(user_id)
        
        # 确保有6个UAV用户，将他们的信道增益保存到状态中
        for i in range(NumberOfUAVUsers):
            if i < len(uav_users):
                user_id = uav_users[i]
                self.State[0,(3*NumberOfUAVs)+i] = User_Channel_Gain_copy[user_id].T
            else:
                # 理论上不应该出现这种情况，因为UAV用户数量固定为6个
                self.State[0,(3*NumberOfUAVs)+i] = 0.0

        Stat_for_return = copy.deepcopy(self.State)
        return Stat_for_return

    def take_action_NOMA(self,action_number,acting_UAV,User_asso_list,ChannelGain_list):
        UAV_move_direction = action_number % 7  #UAV has seven positional actions
        if UAV_move_direction == 0:# UAV moves along the positive half axis of the x-axis
            self.PositionOfUAVs.iloc[0,acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[0,acting_UAV] > self.Zone_border_X:
                self.PositionOfUAVs.iloc[0, acting_UAV] = self.Zone_border_X
        elif UAV_move_direction == 1: # UAV moves along the negative half axis of the x-axis
            self.PositionOfUAVs.iloc[0, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[0, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[0, acting_UAV] = 0
        elif UAV_move_direction == 2: # UAV moves along the positive half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] > self.Zone_border_Y:
                self.PositionOfUAVs.iloc[1, acting_UAV] = self.Zone_border_Y
        elif UAV_move_direction == 3: # UAV moves along the negative half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[1, acting_UAV] = 0
        elif UAV_move_direction == 4: # UAV moves along the positive half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] > self.Zone_border_Z:
                self.PositionOfUAVs.iloc[2, acting_UAV] = self.Zone_border_Z
        elif UAV_move_direction == 5: # UAV moves along the negative half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] < 20:
                self.PositionOfUAVs.iloc[2, acting_UAV] = 20
        elif UAV_move_direction == 6: # UAV hold the position
            pass

        # Power allocation part - NOMA
        power_allocation_scheme = action_number//7  # decode the power allocation action,
        acting_user_list = np.where(User_asso_list == acting_UAV)[0]
        # 获取分配给该UAV的用户（不再限制为前6个）
        if len(acting_user_list) >= 2:
            First_user = acting_user_list[0]
            Second_user = acting_user_list[1]
        else:
            # 如果没有足够的UAV用户，跳过功率分配
            return

        # SIC decoding order
        first_user_CG = ChannelGain_list.iloc[0,First_user]
        second_user_CG = ChannelGain_list.iloc[0,Second_user]
        if first_user_CG >= second_user_CG:
            User0 = Second_user
            User1 = First_user
        else:
            User0 = First_user
            User1 = Second_user

        # three power levels for each user
        # for the weak user, the power levels can be 2, 4, 7 * power unit
        if power_allocation_scheme % UAV_POWER_LEVELS == 0:
            self.Power_allocation_list.iloc[0,User0] = self.Power_unit*2
        elif power_allocation_scheme % UAV_POWER_LEVELS == 1:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*4
        elif power_allocation_scheme % UAV_POWER_LEVELS == 2:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*7
        # for the strong user, the power levels can be 1, 1/2, 1/4 * power unit
        if power_allocation_scheme // UAV_POWER_LEVELS == 0:
            self.Power_allocation_list.iloc[0,User1] = self.Power_unit
        elif power_allocation_scheme // UAV_POWER_LEVELS == 1:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/2
        elif power_allocation_scheme // UAV_POWER_LEVELS == 2:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/4

    def take_action_BS(self, action_number, user_association_list):
        """BS执行功率分配动作（NOMA系统）"""
        # 获取BS用户列表
        bs_users = np.where(user_association_list == NumberOfUAVs)[0]
        
        if len(bs_users) < 2:
            return  # 至少需要2个用户进行NOMA
        
        # 解码动作：将action_number转换为6个用户的功率分配（4^6个组合）
        power_allocations = []
        temp_action = action_number
        
        for i in range(BSUserNumber):
            power_level = temp_action % BS_POWER_LEVELS
            power_allocations.append(power_level)
            temp_action //= BS_POWER_LEVELS
        
        # 根据信道增益确定SIC解码顺序
        bs_channel_gains = []
        for user_id in bs_users:
            # 获取BS到该用户的信道增益
            bs_distance = np.linalg.norm(self.bs_position - self.PositionOfUsers.iloc[:, user_id])
            # 使用BS频段计算路径损耗（含频率修正项）
            freq_factor_db = 20 * math.log10(max(BS_FREQ / 2.0, 1e-6))
            PL = 128.1 + 37.6 * math.log10(bs_distance / 1000) + freq_factor_db
            channel_gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-PL / 10))
            bs_channel_gains.append((user_id, channel_gain))
        
        # 按信道增益排序（强用户在前，弱用户在后）
        bs_channel_gains.sort(key=lambda x: x[1], reverse=True)
        
        # 分配功率：弱用户获得高功率，强用户获得低功率（最大倍数限定为7，保证Pmax_BS=20*Pmax_UAV）
        for i, (user_id, _) in enumerate(bs_channel_gains):
            power_level = power_allocations[i]
            # 功率等级：0,1,2,3 -> 倍数 1, 2, 4, 7
            if power_level == 0:
                power_multiplier = 1
            elif power_level == 1:
                power_multiplier = 2
            elif power_level == 2:
                power_multiplier = 4
            elif power_level == 3:
                power_multiplier = 7
            
            self.Power_allocation_list.iloc[0, user_id] = BS_power_unit * power_multiplier

    def calculate_uav_reward(self, uav_id, user_association_list, data_rate_list, total_sum_rate, episode=-1, step=-1):
        """
        Calculate individual UAV reward based on its served users' QoS requirements
        Args:
            uav_id: ID of the UAV
            user_association_list: User association information
            data_rate_list: Data rate for each user
            total_sum_rate: Total system sum rate
            episode: Current episode number
            step: Current time step
        Returns:
            uav_reward: Individual UAV reward
        """
        # Find users served by this UAV
        served_users = np.where(user_association_list == uav_id)[0]
        
        if len(served_users) == 0:
            return 0.0  # 如果没有服务用户，返回0奖励
        
        # Calculate UAV's sum rate and QoS metrics
        uav_sum_rate = 0
        qos_violation_count = 0
        qos_satisfaction_rates = []
        
        for user_id in served_users:
            user_rate = data_rate_list.iloc[0, user_id]
            uav_sum_rate += user_rate
            
            # Calculate QoS satisfaction rate (0-1)
            qos_satisfaction = min(user_rate / R_require, 1.0)
            qos_satisfaction_rates.append(qos_satisfaction)
            
            # Check if this user meets QoS requirement
            if user_rate < R_require:
                qos_violation_count += 1
        
        # 改进的QoS惩罚机制
        if qos_violation_count == 0:
            # 所有用户都满足QoS，给予额外奖励
            qos_bonus = 1.2  # 增加QoS满足奖励到20%
            uav_reward = uav_sum_rate * qos_bonus
            violation_ratio = 0.0
            avg_qos_satisfaction = 1.0
            penalty_factor = 0.0
        else:
            # 渐进式惩罚：根据违反QoS的用户比例和平均满足度
            violation_ratio = qos_violation_count / len(served_users)
            avg_qos_satisfaction = np.mean(qos_satisfaction_rates)
            
            # 降低惩罚强度，避免reward过低
            # 惩罚系数：基于违反比例和平均满足度
            penalty_factor = 0.1 + 0.2 * violation_ratio + 0.3 * (1 - avg_qos_satisfaction)
            penalty_factor = min(penalty_factor, 0.5)  # 最大惩罚降低到50%
            
            uav_reward = uav_sum_rate * (1 - penalty_factor)
        
        # 记录QoS违反数据（用于调试）
        self.record_qos_violation_data(episode, step, uav_id, served_users, data_rate_list.values[0], 
                                      qos_violation_count, violation_ratio, avg_qos_satisfaction, 
                                      penalty_factor, uav_reward)
        
        return uav_reward

    def calculate_bs_reward(self, user_association_list, data_rate_list, episode=-1, step=-1):
        """
        计算BS奖励，基于其服务用户的QoS要求
        Args:
            user_association_list: 用户分配信息
            data_rate_list: 每个用户的数据率
            episode: 当前episode编号
            step: 当前时间步
        Returns:
            bs_reward: BS系统奖励
        """
        # 找到BS服务的用户
        bs_users = np.where(user_association_list == NumberOfUAVs)[0]
        
        if len(bs_users) == 0:
            return 0.0  # 如果没有服务用户，返回0奖励
        
        # 计算BS的sum rate和QoS指标
        bs_sum_rate = 0
        qos_violation_count = 0
        qos_satisfaction_rates = []
        
        for user_id in bs_users:
            user_rate = data_rate_list.iloc[0, user_id]
            bs_sum_rate += user_rate
            
            # 计算QoS满足率 (0-1)
            qos_satisfaction = min(user_rate / R_require, 1.0)
            qos_satisfaction_rates.append(qos_satisfaction)
            
            # 检查用户是否满足QoS要求
            if user_rate < R_require:
                qos_violation_count += 1
        
        # 改进的BS奖励机制 - 增强梯度信号
        if qos_violation_count == 0:
            # 所有用户都满足QoS，给予额外奖励
            qos_bonus = 1.2  # 20%的额外奖励
            # 添加数据率平方项，鼓励更高的数据率
            bs_reward = bs_sum_rate * qos_bonus + 0.1 * (bs_sum_rate ** 2) / 1000
            violation_ratio = 0.0
            avg_qos_satisfaction = 1.0
            penalty_factor = 0.0
        else:
            # 渐进式惩罚：根据违反QoS的用户比例和平均满足度
            violation_ratio = qos_violation_count / len(bs_users)
            avg_qos_satisfaction = np.mean(qos_satisfaction_rates)
            
            # 更细粒度的惩罚机制
            penalty_factor = 0.1 + 0.4 * violation_ratio + 0.3 * (1 - avg_qos_satisfaction)
            penalty_factor = min(penalty_factor, 0.7)  # 最大惩罚70%
            
            # 基础奖励
            base_reward = bs_sum_rate * (1 - penalty_factor)
            # 添加QoS满足度奖励，鼓励改善QoS
            qos_improvement_bonus = avg_qos_satisfaction * 10
            # 添加数据率奖励，鼓励提高数据率
            data_rate_bonus = 0.05 * bs_sum_rate
            
            bs_reward = base_reward + qos_improvement_bonus + data_rate_bonus
        
        # 记录BS QoS违反数据（用于调试）
        self.record_qos_violation_data(episode, step, NumberOfUAVs, bs_users, data_rate_list.values[0], 
                                      qos_violation_count, violation_ratio, avg_qos_satisfaction, 
                                      penalty_factor, bs_reward)
        
        return bs_reward


class DQN(object):
    def __init__(self):
        self.update_freq = 600  # Model update frequency of the target network
        self.replay_size = REPLAY_BUFFER_SIZE  # 使用全局参数
        self.step = 0
        self.replay_queue = deque(maxlen=self.replay_size)
        self.exploration_phase = True  # 探索阶段标志

        self.power_number = UAV_POWER_LEVELS ** UserNumberPerCell # UAV功率动作数
        self.action_number = 7 * self.power_number # 7个位置动作 × 功率动作数

        # PyTorch model creation
        STATE_DIM = NumberOfUAVs*3 + NumberOfUAVUsers # input layer dim (UAV位置9 + UAV用户信道增益6)
        ACTION_DIM = 7 * self.power_number # output layer dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_model = DQNModel(STATE_DIM, ACTION_DIM).to(self.device)
        
        # Copy weights from model to target model
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def Choose_action(self, s, epsilon, expert_prob=0.0, env=None, uav_id=None, user_association_list=None, episode=-1, step=-1):
        """
        UAV动作选择：结合专家引导和ε-贪心策略
        Args:
            s: 当前状态
            epsilon: ε-贪心参数
            expert_prob: 专家引导概率
            env: 环境对象
            uav_id: UAV ID
            user_association_list: 用户分配列表
            episode: 当前episode
            step: 当前时间步
        Returns:
            action: 选择的动作
        """
        # 探索阶段：纯随机探索
        if self.exploration_phase:
            return np.random.choice(self.action_number)
        
        # 正常训练阶段：专家引导 + ε-贪心
        # 第一层概率选择：专家引导 vs ε-贪心
        if np.random.uniform() < expert_prob:
            # 选择专家引导动作
            return self.expert_guided_action(env, uav_id, user_association_list, episode, step)
        else:
            # 选择ε-贪心动作
            if np.random.uniform() < epsilon:
                return np.random.choice(self.action_number)
            else:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(s).to(self.device)
                    q_values = self.model(s_tensor)
                    return q_values.argmax().item()

    def choose_action_detailed(self, s, epsilon, expert_prob=0.0, env=None, uav_id=None, user_association_list=None, episode=-1, step=-1, topk=5):
        """返回UAV动作与详细诊断信息：专家/探索/贪心、随机数、Top-K Q值，以及动作解码。"""
        info = {
            'mode': '',
            'rand_expert': None,
            'rand_eps': None,
            'topk_indices': [],
            'topk_qvalues': [],
            'move_idx': None,
            'power_combo_idx': None
        }
        r1 = float(np.random.uniform())
        info['rand_expert'] = r1
        if r1 < float(expert_prob):
            info['mode'] = 'expert'
            a = int(self.expert_guided_action(env, uav_id, user_association_list, episode, step))
        else:
            r2 = float(np.random.uniform())
            info['rand_eps'] = r2
            if r2 < float(epsilon):
                info['mode'] = 'explore'
                a = int(np.random.choice(self.action_number))
            else:
                info['mode'] = 'greedy'
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(s).to(self.device)
                    q_values = self.model(s_tensor)
                    q_vec = q_values.squeeze(0) if q_values.dim() > 1 else q_values
                    k = min(int(topk), int(self.action_number))
                    topk_vals, topk_idx = torch.topk(q_vec, k)
                    info['topk_indices'] = [int(i) for i in topk_idx.view(-1).tolist()]
                    info['topk_qvalues'] = [float(v) for v in topk_vals.view(-1).tolist()]
                    a = int(q_vec.argmax().item())
        # 解码UAV复合动作：7个位置动作 × 功率组合
        try:
            # 正确的编码为: action = movement_action + power_action * 7
            info['move_idx'] = int(a % 7)
            info['power_combo_idx'] = int(a // 7)
        except Exception:
            pass
        return a, info

    def expert_guided_action(self, env, uav_id, user_association_list, episode=-1, step=-1):
        """
        专家引导动作决策
        直接从环境获取位置信息，不干扰DQN训练
        """
        # 1. 直接从环境获取UAV位置
        uav_position = np.array([
            env.PositionOfUAVs.iloc[0, uav_id],  # x
            env.PositionOfUAVs.iloc[1, uav_id],  # y  
            env.PositionOfUAVs.iloc[2, uav_id]   # z
        ])
        
        # 2. 获取当前UAV服务的所有用户（不进行UAV用户过滤）
        served_users = np.where(user_association_list == uav_id)[0]
        
        # 3. 获取用户位置
        user_positions = []
        for user_id in served_users:
            user_pos = np.array([
                env.PositionOfUsers.iloc[0, user_id],  # x
                env.PositionOfUsers.iloc[1, user_id],  # y
                env.PositionOfUsers.iloc[2, user_id]   # z (应该是0)
            ])
            user_positions.append(user_pos)
        
        # 4. 计算中点
        if len(user_positions) == 2:
            midpoint = (user_positions[0] + user_positions[1]) / 2
        elif len(user_positions) == 1:
            # 如果只有一个用户，直接使用用户位置
            midpoint = user_positions[0]
        else:
            # 如果没有用户，保持当前位置
            return 6  # 保持位置动作
        
        # 5. 应用高度约束
        min_height = 20
        if midpoint[2] < min_height:
            midpoint[2] = min_height
        
        # 6. 为UAV2提供特殊的专家引导策略
        if uav_id == 2:  # UAV2特殊处理
            # 如果UAV2距离用户太远，增加移动步长
            distance_to_midpoint = np.linalg.norm(uav_position - midpoint)
            if distance_to_midpoint > 100:  # 距离超过100m时
                # 使用更激进的移动策略
                movement_action = self.determine_aggressive_movement_action(uav_position, midpoint)
            else:
                movement_action = self.determine_movement_action(uav_position, midpoint)
        else:
            # 其他UAV使用标准策略
            movement_action = self.determine_movement_action(uav_position, midpoint)
        
        # 7. 使用贪心策略选择Q值最大的功率分配动作
        power_action = self.get_best_power_action(env, uav_id, user_association_list)
        
        # 8. 组合动作
        expert_action = self.combine_actions(movement_action, power_action)
        
        # 9. 记录专家引导数据（用于调试）
        distance_to_midpoint = np.linalg.norm(uav_position - midpoint)
        
        env.record_expert_guidance_data(episode, step, uav_id, uav_position, served_users, 
                                       user_positions, midpoint, expert_action, distance_to_midpoint)
        
        return expert_action

    def determine_movement_action(self, current_pos, target_pos):
        """
        根据当前位置和目标位置确定移动动作
        考虑5米距离阈值避免震荡
        """
        # 计算3D距离
        distance = np.linalg.norm(current_pos - target_pos)
        
        # 距离阈值控制
        if distance < 5.0:
            return 6  # 保持位置，避免震荡
        
        # 计算各方向的距离差
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        
        # 选择距离差最大的方向进行移动
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)
        
        if abs_dx >= abs_dy and abs_dx >= abs_dz:
            return 0 if dx > 0 else 1  # +X 或 -X
        elif abs_dy >= abs_dx and abs_dy >= abs_dz:
            return 2 if dy > 0 else 3  # +Y 或 -Y
        else:
            return 4 if dz > 0 else 5  # +Z 或 -Z

    def determine_aggressive_movement_action(self, current_pos, target_pos):
        """
        为UAV2提供的激进移动策略
        当距离较远时，允许更激进的移动
        """
        # 计算3D距离
        distance = np.linalg.norm(current_pos - target_pos)
        
        # 距离阈值控制（降低阈值，允许更激进的移动）
        if distance < 3.0:  # 降低到3米
            return 6  # 保持位置
        
        # 计算各方向的距离差
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        
        # 选择距离差最大的方向进行移动
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)
        
        if abs_dx >= abs_dy and abs_dx >= abs_dz:
            return 0 if dx > 0 else 1  # +X 或 -X
        elif abs_dy >= abs_dx and abs_dy >= abs_dz:
            return 2 if dy > 0 else 3  # +Y 或 -Y
        else:
            return 4 if dz > 0 else 5  # +Z 或 -Z

    def get_best_power_action(self, env, uav_id, user_association_list):
        """
        使用贪心策略选择Q值最大的功率分配动作
        为每个可能的功率分配计算Q值，选择最大的
        """
        # 获取当前状态
        current_state = env.Create_state_Noposition(uav_id, user_association_list, env.ChannelGain_list)
        
        # 获取所有可能的功率分配动作的Q值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state[0]).to(self.device)
            q_values = self.model(state_tensor)
        
        # 对于专家引导，我们只考虑功率分配部分
        # 功率分配动作范围：0 到 power_number-1
        best_power_action = 0
        best_q_value = float('-inf')
        
        # 遍历所有可能的功率分配动作
        for power_action in range(self.power_number):
            # 计算该功率分配动作对应的Q值
            # 由于我们的动作编码是 movement_action + power_action * 7
            # 我们需要为每个功率动作计算所有移动动作的Q值，然后选择最大的
            max_q_for_power = float('-inf')
            
            for movement_action in range(7):  # 7个移动动作
                action = movement_action + power_action * 7
                q_value = q_values[action].item()
                if q_value > max_q_for_power:
                    max_q_for_power = q_value
            
            # 如果这个功率动作的Q值更高，更新最佳选择
            if max_q_for_power > best_q_value:
                best_q_value = max_q_for_power
                best_power_action = power_action
        
        return best_power_action

    def combine_actions(self, movement_action, power_action):
        """
        组合移动动作和功率动作
        movement_action: 0-6 (7个位置动作)
        power_action: 0-8 (9个功率组合)
        """
        # 动作编码：movement_action + power_action * 7
        return movement_action + power_action * 7

    def remember(self, s, a, next_s, reward):
        # save MDP transitions
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=BATCH_SIZE, lr=1, factor=1):
        # 检查是否完成探索阶段
        if self.exploration_phase and len(self.replay_queue) >= self.replay_size:
            self.exploration_phase = False
            print(f"UAV探索阶段完成，开始正常训练。经验回放区大小: {len(self.replay_queue)}")
        
        # 在探索阶段不进行训练
        if self.exploration_phase:
            return
        
        if len(self.replay_queue) < batch_size:
            return # 缓冲区未满时不训练
        self.step += 1

        # Over 'update_freq' steps, assign the weight of the model to the target_model
        if self.step % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = torch.FloatTensor([replay[0] for replay in replay_batch]).to(self.device)
        next_s_batch = torch.FloatTensor([replay[2] for replay in replay_batch]).to(self.device)

        Q = self.model(s_batch) # calculate Q value
        Q_next = self.target_model(next_s_batch) # predict Q value

        # Create target Q values
        target_Q = Q.clone()
        
        # update Q value following bellman function
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            target_Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * torch.max(Q_next[i]))

        # DNN training
        self.optimizer.zero_grad()
        loss = self.criterion(Q, target_Q)
        loss.backward()
        self.optimizer.step()

class BSDQN(object):
    """BS专用的DQN类，只进行功率分配训练"""
    def __init__(self):
        self.update_freq = 300  # 降低目标网络更新频率，提高训练频率
        self.replay_size = REPLAY_BUFFER_SIZE  # 使用全局参数
        self.step = 0
        self.replay_queue = deque(maxlen=self.replay_size)
        self.exploration_phase = True  # 探索阶段标志

        # BS只进行功率分配，6个用户 × 4个功率等级
        self.power_levels = BS_POWER_LEVELS
        self.action_number = self.power_levels ** BSUserNumber  # 4^6 = 4096个动作

        # PyTorch model creation - 增强网络结构
        STATE_DIM = BSUserNumber  # BS状态空间：6个BS用户的信道增益
        ACTION_DIM = self.action_number  # BS动作空间：6个用户的功率分配组合
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使用更深的网络结构
        self.model = self.create_enhanced_model(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_model = self.create_enhanced_model(STATE_DIM, ACTION_DIM).to(self.device)
        
        # Copy weights from model to target model
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer - 提高学习率
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)  # 提高学习率
        self.criterion = nn.MSELoss()
    
    def create_enhanced_model(self, state_dim, action_dim):
        """创建增强的BS DQN模型"""
        class EnhancedBSDQN(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(EnhancedBSDQN, self).__init__()
                self.fc1 = nn.Linear(state_dim, 64)
                self.fc2 = nn.Linear(64, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, action_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)  # 添加dropout防止过拟合
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        
        return EnhancedBSDQN(state_dim, action_dim)

    def Choose_action(self, s, epsilon):
        """BS动作选择：ε-贪心策略（简版，兼容旧调用）"""
        # 探索阶段：纯随机探索
        if self.exploration_phase:
            return np.random.choice(self.action_number)
        
        # 正常训练阶段：ε-贪心策略
        if np.random.uniform() < epsilon:
            return np.random.choice(self.action_number)
        else:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(s).to(self.device)
                q_values = self.model(s_tensor)
                return q_values.argmax().item()

    def choose_action_detailed(self, s, epsilon, topk=5):
        """返回动作与详细诊断信息：探索/贪心、随机数、Top-K Q值等"""
        info = {
            'mode': '',
            'rand': None,
            'topk_indices': [],
            'topk_qvalues': []
        }
        
        # 探索阶段：纯随机探索
        if self.exploration_phase:
            info['mode'] = 'exploration_phase'
            a = int(np.random.choice(self.action_number))
            return a, info
        
        # 正常训练阶段：ε-贪心策略
        r = float(np.random.uniform())
        info['rand'] = r
        if r < float(epsilon):
            info['mode'] = 'explore'
            a = int(np.random.choice(self.action_number))
            return a, info
        info['mode'] = 'greedy'
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s).to(self.device)
            q_values = self.model(s_tensor)
            q_vec = q_values.squeeze(0) if q_values.dim() > 1 else q_values
            # Top-K
            k = min(int(topk), int(self.action_number))
            topk_vals, topk_idx = torch.topk(q_vec, k)
            info['topk_indices'] = [int(i) for i in topk_idx.view(-1).tolist()]
            info['topk_qvalues'] = [float(v) for v in topk_vals.view(-1).tolist()]
            a = int(q_vec.argmax().item())
            return a, info

    def remember(self, s, a, next_s, reward):
        """保存MDP转换"""
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=BATCH_SIZE, lr=1, factor=1):
        """BS DQN训练"""
        # 检查是否完成探索阶段
        if self.exploration_phase and len(self.replay_queue) >= self.replay_size:
            self.exploration_phase = False
            print(f"BS探索阶段完成，开始正常训练。经验回放区大小: {len(self.replay_queue)}")
        
        # 在探索阶段不进行训练
        if self.exploration_phase:
            return
        
        if len(self.replay_queue) < batch_size:
            return  # 缓冲区未满时不训练
        
        self.step += 1
        
        # 添加调试信息：每100步打印一次训练状态
        if self.step % 100 == 0:
            print(f"BS训练状态: step={self.step}, replay_size={len(self.replay_queue)}")

        # 定期更新目标网络
        if self.step % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = torch.FloatTensor([replay[0] for replay in replay_batch]).to(self.device)
        next_s_batch = torch.FloatTensor([replay[2] for replay in replay_batch]).to(self.device)

        Q = self.model(s_batch)
        Q_next = self.target_model(next_s_batch)

        # 创建目标Q值
        target_Q = Q.clone()
        
        # 更新Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            target_Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * torch.max(Q_next[i]))

        # DNN训练
        self.optimizer.zero_grad()
        loss = self.criterion(Q, target_Q)
        loss.backward()
        self.optimizer.step()




def create_episode_gif(uav_positions_history, user_positions_history, user_association_history, power_allocation_history, episode_num, gif_folder, bs_positions_history=None):
    """Create GIF animation for an episode showing UAV, BS and user movements and power levels"""
    
    # Colors for different UAVs
    uav_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    bs_color = 'black'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def animate(frame):
        ax.clear()
        
        # Set plot limits
        ax.set_xlim(0, ServiceZone_X)
        ax.set_ylim(0, ServiceZone_Y)
        ax.set_title(f'Episode {episode_num} - Time Step {frame+1} (Dynamic User Assignment: Colored=UAV, Black=BS)', fontsize=14)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        
        # Plot BS (star shape)
        if bs_positions_history and frame < len(bs_positions_history):
            bs_positions = bs_positions_history[frame]
            bs_x, bs_y, bs_z = bs_positions.flatten()
            
            # Draw star for BS
            star = plt.Polygon([
                [bs_x, bs_y+15], [bs_x-5, bs_y+5], [bs_x-10, bs_y+5], 
                [bs_x-5, bs_y-5], [bs_x, bs_y-15], [bs_x+5, bs_y-5], 
                [bs_x+10, bs_y+5], [bs_x+5, bs_y+5]
            ], facecolor=bs_color, edgecolor='black', linewidth=2, alpha=0.9)
            ax.add_patch(star)
            
            # Add BS label
            ax.text(bs_x, bs_y+25, 'BS', ha='center', va='bottom', fontweight='bold', fontsize=12, color=bs_color)
            ax.text(bs_x+15, bs_y, f'z={bs_z:.0f}m', ha='left', va='center', fontsize=8, color=bs_color)
        
        # Plot UAVs as triangles
        uav_positions = uav_positions_history[frame]
        user_positions = user_positions_history[frame]
        # 安全索引，防止越界
        idx = min(frame, len(user_association_history)-1, len(power_allocation_history)-1)
        user_asso = user_association_history[idx]
        power_alloc = power_allocation_history[idx]
        uav_power_levels = []
        for uav in range(NumberOfUAVs):
            user_idx = np.where(user_asso == uav)[0]
            if len(user_idx) >= 2:
                p1 = power_alloc.iloc[0, user_idx[0]]
                p2 = power_alloc.iloc[0, user_idx[1]]
                p1_level = round(p1 / UAV_power_unit)
                p2_level = round(p2 / UAV_power_unit)
                uav_power_levels.append((p1_level, p2_level))
            else:
                uav_power_levels.append((0, 0))
        
        for i in range(NumberOfUAVs):
            x, y, z = uav_positions[:, i]
            color = uav_colors[i % len(uav_colors)]
            # Draw triangle for UAV
            triangle = plt.Polygon([[x-10, y-10], [x+10, y-10], [x, y+10]], 
                                  facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(triangle)
            # Add UAV label
            ax.text(x, y+15, f'UAV{i}', ha='center', va='bottom', fontweight='bold')
            # Add height label
            ax.text(x+15, y, f'z={z:.0f}m', ha='left', va='center', fontsize=8)
            p1, p2 = uav_power_levels[i]
            ax.text(x, y-25, f'P1:{p1}  P2:{p2}', ha='center', va='top', fontsize=10, color=color, fontweight='bold')
        
        # Plot users as circles with colors based on association
        for i in range(TotalUsers):
            x, y, z = user_positions[:, i]
            # Determine which UAV/BS this user belongs to
            if user_asso[i] < NumberOfUAVs:  # UAV用户
                uav_idx = user_asso[i]
                color = uav_colors[uav_idx % len(uav_colors)]
                # Add user label
                ax.text(x, y-10, f'U{i}', ha='center', va='top', fontsize=8)
            else:  # BS用户
                color = bs_color
                # 获取BS用户的功率等级
                bs_power_level = 0
                if i < power_alloc.shape[1]:  # 安全检查
                    user_power = power_alloc.iloc[0, i]
                    if user_power > 0:
                        # 计算功率等级 (1, 2, 4, 7, 10)
                        if user_power == BS_power_unit:
                            bs_power_level = 1
                        elif user_power == BS_power_unit * 2:
                            bs_power_level = 2
                        elif user_power == BS_power_unit * 4:
                            bs_power_level = 3
                        elif user_power == BS_power_unit * 7:
                            bs_power_level = 4
                        elif user_power == BS_power_unit * 10:
                            bs_power_level = 5
                
                # Add user label with power level
                ax.text(x, y-10, f'U{i}(P{bs_power_level})', ha='center', va='top', fontsize=8, color=bs_color, fontweight='bold')
            
            # Draw circle for user
            circle = Circle((x, y), 5, facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
            ax.add_patch(circle)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(uav_positions_history), 
                                  interval=200, repeat=False)
    
    # Save as GIF
    gif_filename = os.path.join(gif_folder, f'episode_{episode_num}_trajectory.gif')
    anim.save(gif_filename, writer='pillow', fps=5)
    plt.close()
    
    print(f"GIF saved: {gif_filename}")


def setup_output_folders():
    """Create timestamped output folders"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main result folder
    result_folder = f"result_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    # Create gif folder
    gif_folder = os.path.join(result_folder, f"gif_{timestamp}")
    os.makedirs(gif_folder, exist_ok=True)
    
    # Create output folder for final plots
    output_folder = os.path.join(result_folder, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    return result_folder, gif_folder, output_folder


def run_once():
    # 使用全局参数
    Episodes_number = TOTAL_EPISODES
    Test_episodes_number = TEST_EPISODES
    T = 60 #total time slots (steps)
    # 每个episode仅在开始进行一次分配
    T_AS = np.array([0])

    # Setup output folders
    result_folder, gif_folder, output_folder = setup_output_folders()
    append_runtime_log(output_folder, f"Start run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        append_runtime_log(output_folder, f"Config: TOTAL_EPISODES={TOTAL_EPISODES}, EXPERT_EPISODES={EXPERT_EPISODES}, TEST_EPISODES={TEST_EPISODES}, UAV_POWER_LEVELS={UAV_POWER_LEVELS}, BS_POWER_LEVELS={BS_POWER_LEVELS}, R_require={R_require}")
    except Exception:
        pass
    # 一次性记录BS动作空间信息（用于排查动作范围异常）
    try:
        append_runtime_log(output_folder, f"BS action space levels={BS_POWER_LEVELS}, expected range=[0,{BS_POWER_LEVELS-1}]")
    except Exception:
        pass
    # 精简启动信息到日志
    append_runtime_log(output_folder, f"Output folders ready: result={result_folder}, gif={gif_folder}, output={output_folder}")
    append_runtime_log(output_folder, f"Train config: total={TOTAL_EPISODES}, expert_ep={EXPERT_EPISODES}, training_ep={TRAINING_EPISODES-EXPERT_EPISODES}, test_ep={TEST_EPISODES}")
    append_runtime_log(output_folder, f"Replay buffer config: size={REPLAY_BUFFER_SIZE}, batch_size={BATCH_SIZE}")
    append_runtime_log(output_folder, f"Exploration phase: {EXPLORATION_EPISODES} episodes needed to fill buffer ({EXPLORATION_EPISODES/TOTAL_EPISODES*100:.1f}% of total episodes)")
    
    # 打印探索阶段信息
    print(f"\n=== 训练配置信息 ===")
    print(f"总训练轮次: {TOTAL_EPISODES}")
    print(f"经验回放区大小: {REPLAY_BUFFER_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"探索阶段: 需要 {EXPLORATION_EPISODES} 个episode 填满经验回放区")
    print(f"探索阶段占比: {EXPLORATION_EPISODES/TOTAL_EPISODES*100:.1f}%")
    print(f"正常训练阶段: {TOTAL_EPISODES - EXPLORATION_EPISODES} 个episode")
    print(f"==================\n")
    
    print(f"开始探索阶段：收集 {REPLAY_BUFFER_SIZE} 个经验...")
    print(f"探索阶段将运行 {EXPLORATION_EPISODES} 个episode，不进行模型训练")
    print(f"探索完成后将开始正式训练...\n")

    env = SystemModel() # crate an environment
    uav_agent = DQN() # crate UAV agent
    bs_agent = BSDQN() # crate BS agent

    Epsilon = 0.9
    BS_Epsilon = 0.9  # BS的ε值
    
    # 数据记录数组（系统/UAV口径：每时隙总和）
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate (system total per step)
    WorstuserRate_seq = np.zeros(T) # Initialize memory to store worst user rate per step (system)
    # 分系统按时隙记录（用于计算每个episode的平均数据率趋势）
    uav_datarate_seq = np.zeros(T)  # sum over UAV-served users per step
    bs_datarate_seq = np.zeros(T)   # sum over BS-served users per step
    bs_worstuser_seq = np.zeros(T)
    
    # 分别记录UAV和BS的性能
    UAV_Through_put_seq = np.zeros(Episodes_number) # UAV系统吞吐量
    BS_Through_put_seq = np.zeros(Episodes_number)  # BS系统吞吐量
    System_Through_put_seq = np.zeros(Episodes_number) # 系统总吞吐量
    
    UAV_Worstuser_TP_seq = np.zeros(Episodes_number) # UAV最差用户吞吐量
    BS_Worstuser_TP_seq = np.zeros(Episodes_number)  # BS最差用户吞吐量
    System_Worstuser_TP_seq = np.zeros(Episodes_number) # 系统最差用户吞吐量

    # 每个episode的平均数据率（新增）
    UAV_AvgDataRate_seq = np.zeros(Episodes_number)
    BS_AvgDataRate_seq = np.zeros(Episodes_number)
    
    # 新增：记录每轮次的reward和每回合最后一步的datarate
    UAV_Reward_seq = np.zeros(Episodes_number)  # 每轮次UAV的reward
    BS_Reward_seq = np.zeros(Episodes_number)   # 每轮次BS的reward
    
    UAV_Final_Datarate_seq = np.zeros(Episodes_number)  # 每回合最后一步UAV的datarate
    BS_Final_Datarate_seq = np.zeros(Episodes_number)   # 每回合最后一步BS的datarate
    
    # 新增：记录每个UAV的详细数据
    Individual_UAV_Reward_seq = np.zeros((NumberOfUAVs, Episodes_number))  # 每个UAV每轮次的reward
    Individual_UAV_Final_Datarate_seq = np.zeros((NumberOfUAVs, Episodes_number))  # 每个UAV每回合最后一步的datarate
    
    # 新增：记录每个用户每个episode的平均数据率
    User_AvgDataRate_seq = np.zeros((TotalUsers, Episodes_number))  # 每个用户每个episode的平均数据率
    
    for episode in range(Episodes_number):
        # 检查是否还在探索阶段
        is_exploration_phase = (uav_agent.exploration_phase or bs_agent.exploration_phase)
        
        env.Reset_position()
        user_association_history = []
        power_allocation_history = []
        
        if not is_exploration_phase:
            # 只在正式训练阶段进行ε衰减
            Epsilon -= 0.9 / (TRAINING_EPISODES) # decaying epsilon (只在训练轮次衰减)
            # BS探索：与UAV一致，按episode衰减一次并加下限
            BS_Epsilon = max(0.05, BS_Epsilon - 0.9 / TRAINING_EPISODES)
            
            # 计算专家引导概率（前35%轮次线性衰减）
            expert_prob = max(0.0, 1.0 - episode / EXPERT_EPISODES) if episode < EXPERT_EPISODES else 0.0
            
            # 为UAV2增加额外的专家引导概率，帮助其学习
            uav2_expert_boost = 0.1 if episode < EXPERT_EPISODES else 0.0  # UAV2额外10%的专家引导概率
        else:
            # 探索阶段：固定参数，不进行衰减
            expert_prob = 0.0
            uav2_expert_boost = 0.0
        
        p=0 # punishment counter
        
        # Record initial positions
        env.record_positions()
        
        # 只在正式训练阶段记录详细日志
        if not is_exploration_phase:
            append_runtime_log(output_folder, f"Episode {episode} begin. Epsilon={Epsilon:.3f}, BS_Epsilon={BS_Epsilon:.3f}, expert_prob={expert_prob:.2f}")
        elif episode % 1 == 0:  # 探索阶段每1个episode打印一次进度
            progress = (episode / EXPLORATION_EPISODES) * 100
            print(f"探索阶段进度: Episode {episode}/{EXPLORATION_EPISODES} ({progress:.1f}%)")
        
        # 新增：记录每轮次的累计reward和最后一步的datarate
        episode_uav_reward = 0.0  # 当前episode的UAV累计reward
        episode_bs_reward = 0.0   # 当前episode的BS累计reward
        episode_uav_final_datarate = 0.0  # 当前episode最后一步UAV的datarate
        episode_bs_final_datarate = 0.0   # 当前episode最后一步BS的datarate
        
        # 新增：记录每个UAV的累计reward和最后一步datarate
        individual_uav_rewards = np.zeros(NumberOfUAVs)  # 每个UAV的累计reward
        individual_uav_final_datarates = np.zeros(NumberOfUAVs)  # 每个UAV最后一步的datarate
        # 新增：QoS与SINR统计累计器（按系统分别统计）
        uav_qos_violations_total = 0
        uav_users_eval_count = 0
        uav_sinr_sum_linear = 0.0
        bs_qos_violations_total = 0
        bs_users_eval_count = 0
        bs_sinr_sum_linear = 0.0
        
        for t in range(T):

            if t == 0:
                User_AS_List = env.bs_uav_dynamic_user_assignment() # 每个episode开始时进行一次分配
                # 只在正式训练阶段记录分配结果
                if not is_exploration_phase:
                    try:
                        uav_users_str = "; ".join([f"UAV{i}:{np.where(User_AS_List==i)[0].tolist()}" for i in range(NumberOfUAVs)])
                        bs_users = np.where(User_AS_List==NumberOfUAVs)[0].tolist()
                        append_runtime_log(output_folder, f"Episode {episode} assign -> {uav_users_str}; BS:{bs_users}")
                    except Exception:
                        pass

            for UAV in range(NumberOfUAVs):

                Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, TotalUsers) # Calculate the distance for each UAV-users
                PL_for_CG = env.Get_Propergation_Loss_UAV(Distence_CG,env.PositionOfUAVs,NumberOfUAVs, TotalUsers) # 仅UAV频段
                CG = env.Get_Channel_Gain_NOMA(NumberOfUAVs, TotalUsers, PL_for_CG, User_AS_List,NoisePower) # Calculate the channel gain for each UAV-users
                Eq_CG = env.Get_Channel_Gain_NOMA(NumberOfUAVs, TotalUsers, PL_for_CG, User_AS_List,NoisePower) # Calculate the equivalent channel gain to determine the decoding order

                State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
                # 为UAV2提供额外的专家引导概率
                if UAV == 2:  # UAV2
                    enhanced_expert_prob = min(1.0, expert_prob + uav2_expert_boost)
                else:
                    enhanced_expert_prob = expert_prob
                
                # UAV选择动作（带详细诊断）
                action_name, uav_info = uav_agent.choose_action_detailed(State, Epsilon, enhanced_expert_prob, env, UAV, User_AS_List, episode, t, topk=5)
                try:
                    append_runtime_log(
                        output_folder,
                        f"Ep{episode} t{t} UAV{UAV} action={action_name} mode={uav_info.get('mode')} rand_expert={uav_info.get('rand_expert')} rand_eps={uav_info.get('rand_eps')} move_idx={uav_info.get('move_idx')} power_combo_idx={uav_info.get('power_combo_idx')} topk_idx={uav_info.get('topk_indices')} topk_q={uav_info.get('topk_qvalues')}"
                    )
                except Exception:
                    pass
                env.take_action_NOMA(action_name,UAV,User_AS_List,Eq_CG) # take UAV action in the environment
                # 记录UAV动作的功率分配解码与用户（弱/强）对应关系
                try:
                    served = np.where(User_AS_List == UAV)[0]
                    if len(served) >= 2:
                        first_u, second_u = int(served[0]), int(served[1])
                        first_cg = float(Eq_CG.iloc[0, first_u]) if hasattr(Eq_CG, 'iloc') else 0.0
                        second_cg = float(Eq_CG.iloc[0, second_u]) if hasattr(Eq_CG, 'iloc') else 0.0
                        if first_cg >= second_cg:
                            weak_u, strong_u = second_u, first_u
                        else:
                            weak_u, strong_u = first_u, second_u
                        power_scheme = int(action_name // 7)
                        weak_level_idx = int(power_scheme % UAV_POWER_LEVELS)
                        strong_level_idx = int(power_scheme // UAV_POWER_LEVELS)
                        # 弱用户功率倍数映射: 0->2, 1->4, 2->7
                        weak_mult = [2, 4, 7][weak_level_idx] if 0 <= weak_level_idx < 3 else None
                        # 强用户功率倍数映射: 0->1, 1->0.5, 2->0.25
                        strong_mult = [1.0, 0.5, 0.25][strong_level_idx] if 0 <= strong_level_idx < 3 else None
                        append_runtime_log(
                            output_folder,
                            f"Ep{episode} t{t} UAV{UAV} served={list(map(int,served[:2]))} weak={weak_u} strong={strong_u} weak_mult={weak_mult} strong_mult={strong_mult}"
                        )
                except Exception:
                    pass

                Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, TotalUsers) # after taking actions, calculate the distance again
                P_L = env.Get_Propergation_Loss_UAV(Distence,env.PositionOfUAVs,NumberOfUAVs, TotalUsers) # 仅UAV频段
                SINR=env.Get_SINR_NNOMA(NumberOfUAVs,TotalUsers,P_L,User_AS_List,Eq_CG,NoisePower) # calculate SINR for users
                DataRate,SumRate,WorstuserRate = env.Calcullate_Datarate(SINR, TotalUsers, Bandwidth) # calculate data rate, sum rate and the worstusers data rate
                #print(DataRate,'\nSumrate==',SumRate,'\nWorstuserRate=',WorstuserRate)

                # QoS与SINR（UAV系统口径）逐步统计与日志
                try:
                    uav_user_idx = np.where(User_AS_List < NumberOfUAVs)[0]
                    if len(uav_user_idx) > 0:
                        step_rates = DataRate.values[0][uav_user_idx]
                        step_sinr = SINR.values[0][uav_user_idx] if hasattr(SINR, 'values') else SINR[uav_user_idx]
                        viol = int(np.sum(step_rates < R_require))
                        uav_qos_violations_total += viol
                        uav_users_eval_count += int(len(uav_user_idx))
                        uav_sinr_sum_linear += float(np.sum(step_sinr))
                        avg_sinr_db = 10*np.log10(np.maximum(np.mean(step_sinr), 1e-12))
                        append_runtime_log(output_folder, f"Ep{episode} t{t} UAV_QoS viol={viol}/{len(uav_user_idx)} avgSINR_dB={avg_sinr_db:.2f}")
                except Exception:
                    pass

                # calculate reward based on sum rate and check if users meet the QOS requirement
                # Calculate individual UAV reward based on its served users' QoS
                UAV_Reward = env.calculate_uav_reward(UAV, User_AS_List, DataRate, SumRate, episode, t)
                if UAV_Reward < SumRate:  # If any UAV's users don't meet QoS, increment punishment counter
                    p+=1
                
                # 新增：记录每个UAV的reward和最后一步的datarate
                individual_uav_rewards[UAV] += UAV_Reward
                # 修复：累计当前episode的UAV总reward
                episode_uav_reward += float(UAV_Reward)
                if t == T - 1:  # 最后一步
                    # 计算该UAV服务用户的总数据率
                    uav_final_sum = 0.0
                    for uid in range(TotalUsers):
                        if User_AS_List[uid] == UAV:
                            uav_final_sum += float(DataRate.iloc[0, uid])
                    individual_uav_final_datarates[UAV] = uav_final_sum
                    # 修复：累计当前episode的UAV最终步总数据率（3个UAV求和）
                    episode_uav_final_datarate += uav_final_sum

                CG_next = env.Get_Channel_Gain_NOMA(NumberOfUAVs, TotalUsers, P_L, User_AS_List,NoisePower)  # Calculate the equivalent channel gain for S_{t+1}
                Next_state = env.Create_state_Noposition(UAV,User_AS_List,CG_next) # Generate S_{t+1}

                #copy data for (S_t,A_t,S_t+1,R_t)
                State_for_memory = copy.deepcopy(State[0])
                Action_for_memory = copy.deepcopy(action_name)
                Next_state_for_memory = copy.deepcopy(Next_state[0])
                Reward_for_memory = copy.deepcopy(UAV_Reward)

                uav_agent.remember(State_for_memory, Action_for_memory, Next_state_for_memory, Reward_for_memory) #save the MDP transitions as (S_t,A_t,S_t+1,R_t)
                uav_agent.train() #train the UAV DQN agent
                env.User_randomMove(MAXUserspeed,TotalUsers) # move users
                
                # Record positions after all UAVs and users have moved
                if UAV==(NumberOfUAVs-1):
                    env.record_positions()
                    user_association_history.append(copy.deepcopy(User_AS_List))
                    power_allocation_history.append(env.Power_allocation_list.copy())
                    Rate_during_t = copy.deepcopy(SumRate)
                    datarate_seq[t] = Rate_during_t
                    
                    # 记录UAV系统当前步总数据率（仅UAV服务的用户）
                    uav_sum_step = 0.0
                    uav_user_rates = []
                    for uid in range(TotalUsers):
                        if User_AS_List[uid] < NumberOfUAVs:
                            uav_sum_step += float(DataRate.iloc[0, uid])
                            uav_user_rates.append(float(DataRate.iloc[0, uid]))
                    uav_datarate_seq[t] = uav_sum_step
                    
                    # 计算UAV用户的最差用户数据率
                    if uav_user_rates:
                        WorstuserRate_seq[t] = float(np.min(uav_user_rates))
                    else:
                        WorstuserRate_seq[t] = 0.0
                    
                    # 暂时不记录数据，等BS训练完成后再记录最终数据
                    
                    # BS训练（在UAV训练完成后）
                    # 获取BS状态：6个BS用户的信道增益
                    bs_users = np.where(User_AS_List == NumberOfUAVs)[0]
                    if len(bs_users) >= 2:  # 至少需要2个用户进行NOMA
                        # 计算BS用户的信道增益
                        bs_channel_gains = []
                        for user_id in bs_users:
                            bs_distance = np.linalg.norm(env.bs_position - env.PositionOfUsers.iloc[:, user_id])
                            # 与执行时一致：加入频率修正项（参考 take_action_BS 与 Get_Propergation_Loss_BS）
                            freq_factor_db = 20 * math.log10(max(BS_FREQ / 2.0, 1e-6))
                            PL = 128.1 + 37.6 * math.log10(bs_distance / 1000) + freq_factor_db
                            channel_gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-PL / 10))
                            bs_channel_gains.append(channel_gain)
                        
                        # BS状态：6个用户的信道增益
                        bs_state = np.array(bs_channel_gains)
                        
                        # BS选择动作（带详细诊断）
                        bs_action, bs_info = bs_agent.choose_action_detailed(bs_state, BS_Epsilon, topk=5)
                        
                        # BS执行动作
                        env.take_action_BS(bs_action, User_AS_List)
                        
                        # 重新计算通信指标（包含BS的新功率分配）
                        Distence_BS = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, TotalUsers)
                        P_L_BS = env.Get_Propergation_Loss_BS(Distence_BS, env.PositionOfUAVs, NumberOfUAVs, TotalUsers, BS_FREQ)  # 仅BS频段
                        # 使用BS路径损耗计算对应的信道增益供NOMA排序
                        CG_BS = env.Get_Channel_Gain_NOMA(NumberOfUAVs, TotalUsers, P_L_BS, User_AS_List, NoisePower)
                        SINR_BS = env.Get_SINR_NNOMA(NumberOfUAVs, TotalUsers, P_L_BS, User_AS_List, CG_BS, NoisePower)
                        DataRate_BS, SumRate_BS, WorstuserRate_BS = env.Calcullate_Datarate(SINR_BS, TotalUsers, Bandwidth)
                        # QoS与SINR（BS系统口径）逐步统计与日志
                        try:
                            bs_idx = np.where(User_AS_List == NumberOfUAVs)[0]
                            if len(bs_idx) > 0:
                                step_rates_bs = DataRate_BS.values[0][bs_idx]
                                step_sinr_bs = SINR_BS.values[0][bs_idx] if hasattr(SINR_BS, 'values') else SINR_BS[bs_idx]
                                viol_bs = int(np.sum(step_rates_bs < R_require))
                                bs_qos_violations_total += viol_bs
                                bs_users_eval_count += int(len(bs_idx))
                                bs_sinr_sum_linear += float(np.sum(step_sinr_bs))
                                avg_sinr_bs_db = 10*np.log10(np.maximum(np.mean(step_sinr_bs), 1e-12))
                                append_runtime_log(output_folder, f"Ep{episode} t{t} BS_QoS viol={viol_bs}/{len(bs_idx)} avgSINR_dB={avg_sinr_bs_db:.2f}")
                                
                                # 计算BS用户的最差用户数据率
                                bs_worst_user_rate = float(np.min(step_rates_bs))
                            else:
                                bs_worst_user_rate = 0.0
                        except Exception:
                            bs_worst_user_rate = 0.0
                        # 记录BS系统当前步
                        bs_datarate_seq[t] = float(SumRate_BS)
                        bs_worstuser_seq[t] = float(bs_worst_user_rate)
                        
                        # 计算BS奖励
                        BS_Reward = env.calculate_bs_reward(User_AS_List, DataRate_BS, episode, t)
                        
                        # 新增：记录BS的reward和最后一步的datarate
                        episode_bs_reward += BS_Reward
                        if t == T - 1:  # 最后一步
                            episode_bs_final_datarate = float(SumRate_BS)
                        
                        # BS DQN训练
                        bs_agent.remember(bs_state, bs_action, bs_state, BS_Reward)  # 简化：next_state = current_state
                        bs_agent.train()
                        
                        # 记录BS动作来源与Top-K Q值
                        try:
                            append_runtime_log(
                                output_folder,
                                f"Ep{episode} t{t} BS_action={bs_action} mode={bs_info.get('mode')} rand={bs_info.get('rand'):.4f} topk_idx={bs_info.get('topk_indices')} topk_q={bs_info.get('topk_qvalues')} BS_Reward={BS_Reward:.2f}"
                            )
                        except Exception:
                            pass
                        
                        # 记录BS训练完成后的数据（包含BS的功率分配效果）
                        env.record_episode_data(episode, t, User_AS_List, DataRate_BS, SumRate_BS, expert_prob, Epsilon)
                    else:
                        # 即使BS用户数量不足，也要记录数据（使用UAV训练完成后的数据）
                        env.record_episode_data(episode, t, User_AS_List, DataRate, SumRate, expert_prob, Epsilon)
                        
                        # BS的ε值按episode衰减，已在回合开始时更新
                        # 记录BS组合动作解码为每用户功率等级/倍数/实际功率（W）
                        try:
                            temp_action = int(bs_action)
                            levels = []
                            mults = []
                            powers_w = []
                            for _ in range(BSUserNumber):
                                lvl = int(temp_action % BS_POWER_LEVELS)
                                levels.append(lvl)
                                temp_action //= BS_POWER_LEVELS
                                mult = [1, 2, 4, 7][lvl]
                                mults.append(mult)
                                powers_w.append(float(BS_power_unit * mult))
                            append_runtime_log(output_folder, f"Ep{episode} t{t} BS_levels={levels} BS_mult={mults} BS_powerW={powers_w}")
                        except Exception:
                            pass


        # 计算UAV和BS的性能
        UAV_Through_put = np.sum(datarate_seq) # calculate UAV throughput for an episode
        UAV_Worstuser_TP = np.sum(WorstuserRate_seq) # calculate UAV worst user throughput for an episode
        
        # 计算BS性能（与UAV口径对齐：按时隙累加）
        BS_Through_put = np.sum(bs_datarate_seq)
        BS_Worstuser_TP = np.sum(bs_worstuser_seq)

        # 记录每个episode的平均数据率（新增）
        UAV_AvgDataRate_seq[episode] = float(np.mean(uav_datarate_seq))
        BS_AvgDataRate_seq[episode] = float(np.mean(bs_datarate_seq))
        
        # 新增：记录每轮次的累计reward和最后一步的datarate
        UAV_Reward_seq[episode] = episode_uav_reward
        BS_Reward_seq[episode] = episode_bs_reward
        UAV_Final_Datarate_seq[episode] = episode_uav_final_datarate
        BS_Final_Datarate_seq[episode] = episode_bs_final_datarate
        
        # 新增：记录每个UAV的详细数据
        Individual_UAV_Reward_seq[:, episode] = individual_uav_rewards
        Individual_UAV_Final_Datarate_seq[:, episode] = individual_uav_final_datarates
        
        # 新增：计算并记录每个用户在这个episode的平均数据率
        # 从debug_data中获取这个episode的所有数据
        episode_data = [data for data in env.debug_data['episode_data'] if data['episode'] == episode]
        if episode_data:
            # 为每个用户计算平均数据率
            for user_id in range(TotalUsers):
                user_data_rates = []
                for data in episode_data:
                    if user_id < len(data['data_rates']):
                        user_data_rates.append(data['data_rates'][user_id])
                
                if user_data_rates:
                    User_AvgDataRate_seq[user_id, episode] = np.mean(user_data_rates)
        
        # 系统总性能
        System_Through_put = UAV_Through_put + BS_Through_put
        System_Worstuser_TP = min(UAV_Worstuser_TP, BS_Worstuser_TP) if BS_Worstuser_TP > 0 else UAV_Worstuser_TP
        
        # 保存性能数据
        UAV_Through_put_seq[episode] = UAV_Through_put
        BS_Through_put_seq[episode] = BS_Through_put
        System_Through_put_seq[episode] = System_Through_put
        
        UAV_Worstuser_TP_seq[episode] = UAV_Worstuser_TP
        BS_Worstuser_TP_seq[episode] = BS_Worstuser_TP
        System_Worstuser_TP_seq[episode] = System_Worstuser_TP

        # 新增：UAV2性能监控
        uav2_performance = individual_uav_rewards[2] if len(individual_uav_rewards) > 2 else 0
        uav2_final_datarate = individual_uav_final_datarates[2] if len(individual_uav_final_datarates) > 2 else 0
        
        # 检查探索阶段是否完成
        if is_exploration_phase and not (uav_agent.exploration_phase or bs_agent.exploration_phase):
            print(f"\n=== 探索阶段完成！===")
            print(f"UAV经验回放区大小: {len(uav_agent.replay_queue)}")
            print(f"BS经验回放区大小: {len(bs_agent.replay_queue)}")
            print(f"开始正式训练阶段...")
            print(f"==================\n")
            is_exploration_phase = False
        
        # 只在正式训练阶段打印详细信息
        if not is_exploration_phase:
            print('Episode=',episode,'UAV_Epsilon=',Epsilon,'BS_Epsilon=',BS_Epsilon,'Expert_Prob=',expert_prob,'Punishment=',p,'UAV_TP=',UAV_Through_put,'BS_TP=',BS_Through_put,'System_TP=',System_Through_put)
        elif episode % 1 == 0:  # 探索阶段每1个episode打印一次进度
            progress = (episode / EXPLORATION_EPISODES) * 100
            print(f"探索阶段进度: Episode {episode}/{EXPLORATION_EPISODES} ({progress:.1f}%)")
        # runtime_log增加更详细统计，便于后续诊断
        try:
            per_uav_reward = individual_uav_rewards.tolist()
            per_uav_final = individual_uav_final_datarates.tolist()
        except Exception:
            per_uav_reward = []
            per_uav_final = []
        # 计算每回合平均SINR（线性均值转dB）与QoS违反率
        try:
            uav_avg_sinr_db_ep = 10*np.log10(max(uav_sinr_sum_linear / max(uav_users_eval_count, 1), 1e-12))
            uav_violation_rate_ep = (uav_qos_violations_total / max(uav_users_eval_count, 1)) if uav_users_eval_count > 0 else 0.0
        except Exception:
            uav_avg_sinr_db_ep = 0.0
            uav_violation_rate_ep = 0.0
        try:
            bs_avg_sinr_db_ep = 10*np.log10(max(bs_sinr_sum_linear / max(bs_users_eval_count, 1), 1e-12))
            bs_violation_rate_ep = (bs_qos_violations_total / max(bs_users_eval_count, 1)) if bs_users_eval_count > 0 else 0.0
        except Exception:
            bs_avg_sinr_db_ep = 0.0
            bs_violation_rate_ep = 0.0
        append_runtime_log(
            output_folder,
            f"Episode {episode} summary: UAV_TP={UAV_Through_put:.2f}, BS_TP={BS_Through_put:.2f}, System_TP={System_Through_put:.2f}, p={p}; perUAV_reward={per_uav_reward}, perUAV_final={per_uav_final}; UAV_QoS_ep={uav_qos_violations_total}/{uav_users_eval_count}({uav_violation_rate_ep:.2%}) UAV_avgSINR_dB_ep={uav_avg_sinr_db_ep:.2f}; BS_QoS_ep={bs_qos_violations_total}/{bs_users_eval_count}({bs_violation_rate_ep:.2%}) BS_avgSINR_dB_ep={bs_avg_sinr_db_ep:.2f}"
        )
        
        # Generate GIF for this episode
        if len(env.uav_positions_history) > 0:
            # 只在正式训练阶段保存GIF
            if not is_exploration_phase:
                create_episode_gif(env.uav_positions_history, env.user_positions_history, user_association_history, power_allocation_history, episode, gif_folder, env.bs_positions_history)

    # save data
    np.save(os.path.join(result_folder, "UAV_Through_put_NOMA.npy"), UAV_Through_put_seq)
    np.save(os.path.join(result_folder, "BS_Through_put_NOMA.npy"), BS_Through_put_seq)
    np.save(os.path.join(result_folder, "System_Through_put_NOMA.npy"), System_Through_put_seq)
    np.save(os.path.join(result_folder, "UAV_WorstUser_Through_put_NOMA.npy"), UAV_Worstuser_TP_seq)
    np.save(os.path.join(result_folder, "BS_WorstUser_Through_put_NOMA.npy"), BS_Worstuser_TP_seq)
    np.save(os.path.join(result_folder, "System_WorstUser_Through_put_NOMA.npy"), System_Worstuser_TP_seq)
    np.save(os.path.join(result_folder, "Total Data Rate_NOMA.npy"), datarate_seq)
    np.save(os.path.join(result_folder, "PositionOfUsers_end_NOMA.npy"), env.PositionOfUsers)
    np.save(os.path.join(result_folder, "PositionOfUAVs_end_NOMA.npy"), env.PositionOfUAVs)
    
    # 新增：保存reward和datarate数据
    np.save(os.path.join(result_folder, "UAV_Reward_NOMA.npy"), UAV_Reward_seq)
    np.save(os.path.join(result_folder, "BS_Reward_NOMA.npy"), BS_Reward_seq)
    np.save(os.path.join(result_folder, "UAV_Final_Datarate_NOMA.npy"), UAV_Final_Datarate_seq)
    np.save(os.path.join(result_folder, "BS_Final_Datarate_NOMA.npy"), BS_Final_Datarate_seq)
    np.save(os.path.join(result_folder, "Individual_UAV_Reward_NOMA.npy"), Individual_UAV_Reward_seq)
    np.save(os.path.join(result_folder, "Individual_UAV_Final_Datarate_NOMA.npy"), Individual_UAV_Final_Datarate_seq)
    np.save(os.path.join(result_folder, "User_AvgDataRate_NOMA.npy"), User_AvgDataRate_seq)
    
    # 保存调试数据
    env.save_debug_data(output_folder)
    
    # 生成仿真参数报告
    env.generate_simulation_parameters_report(output_folder, TOTAL_EPISODES, EXPERT_EPISODES, 
                                           TEST_EPISODES, TRAINING_EPISODES, UAV_Through_put_seq, 
                                           BS_Through_put_seq, System_Through_put_seq, 
                                           UAV_Worstuser_TP_seq, BS_Worstuser_TP_seq, 
                                           System_Worstuser_TP_seq, datarate_seq, UAV_Reward_seq, 
                                           BS_Reward_seq, UAV_Final_Datarate_seq, BS_Final_Datarate_seq)

    # Create and save throughput plot (显示3条曲线)
    plt.figure(figsize=(12, 8))
    x_axis = range(1, Episodes_number+1)
    plt.plot(x_axis, System_Through_put_seq, 'b-', linewidth=3, label='System Total', marker='o', markersize=4)
    plt.plot(x_axis, UAV_Through_put_seq, 'r-', linewidth=2, label='UAV System', marker='s', markersize=3)
    plt.plot(x_axis, BS_Through_put_seq, 'g-', linewidth=2, label='BS System', marker='^', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Throughput', fontsize=14)
    plt.title('System Throughput vs Episodes (UAV + BS)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'System_Throughput_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 新增：每个episode的平均数据率趋势（UAV/BS）
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis, UAV_AvgDataRate_seq, 'r-', linewidth=2, label='UAV Avg DataRate per Episode', marker='s', markersize=3)
    plt.plot(x_axis, BS_AvgDataRate_seq, 'g-', linewidth=2, label='BS Avg DataRate per Episode', marker='^', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Data Rate per Episode', fontsize=14)
    plt.title('Average Data Rate Trend per Episode (UAV vs BS)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Avg_DataRate_Per_Episode.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create and save worst user throughput plot (显示3条曲线)
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis, System_Worstuser_TP_seq, 'b-', linewidth=3, label='System Total', marker='o', markersize=4)
    plt.plot(x_axis, UAV_Worstuser_TP_seq, 'r-', linewidth=2, label='UAV System', marker='s', markersize=3)
    plt.plot(x_axis, BS_Worstuser_TP_seq, 'g-', linewidth=2, label='BS System', marker='^', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Throughput of Worst User', fontsize=14)
    plt.title('Worst User Throughput vs Episodes (UAV + BS)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'System_WorstUser_Throughput_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 重写：System_Data_Rate_NOMA - 最后一个episode中每个step的系统数据率
    plt.figure(figsize=(12, 8))
    x_axis_T = range(1, T+1)
    
    # 计算并显示UAV、BS和系统总数据率在最后一个episode的每个step
    if len(env.debug_data['episode_data']) > 0:
        # 获取最后一个episode的数据（最后T个时间步）
        final_episode_data = env.debug_data['episode_data'][-T:] if len(env.debug_data['episode_data']) >= T else []
        if final_episode_data:
            uav_data_rates = []
            bs_data_rates = []
            system_data_rates = []
            
            for data in final_episode_data:
                user_assoc = data['user_association']
                data_rates = data['data_rates']
                
                # 分离UAV和BS用户的数据率
                uav_sum_rate = sum(data_rates[i] for i, assoc in enumerate(user_assoc) if assoc < NumberOfUAVs)
                bs_sum_rate = sum(data_rates[i] for i, assoc in enumerate(user_assoc) if assoc == NumberOfUAVs)
                system_sum_rate = uav_sum_rate + bs_sum_rate
                
                uav_data_rates.append(uav_sum_rate)
                bs_data_rates.append(bs_sum_rate)
                system_data_rates.append(system_sum_rate)
            
            # 绘制三条曲线
            if system_data_rates:
                plt.plot(x_axis_T, system_data_rates, 'b-', linewidth=3, label='System Total (UAV + BS)', marker='o', markersize=4)
            if uav_data_rates:
                plt.plot(x_axis_T, uav_data_rates, 'r-', linewidth=2, label='UAV System', marker='s', markersize=3)
            if bs_data_rates:
                plt.plot(x_axis_T, bs_data_rates, 'g-', linewidth=2, label='BS System', marker='^', markersize=3)
        else:
            # 如果没有debug数据，使用datarate_seq作为系统总数据率
            plt.plot(x_axis_T, datarate_seq, 'b-', linewidth=3, label='System Total (Last Episode)', marker='o', markersize=4)
    else:
        # 如果没有位置历史，使用datarate_seq作为系统总数据率
        plt.plot(x_axis_T, datarate_seq, 'b-', linewidth=3, label='System Total (Last Episode)', marker='o', markersize=4)
    
    plt.xlabel('Steps in Last Episode', fontsize=14)
    plt.ylabel('Data Rate of System', fontsize=14)
    plt.title('System Data Rate vs Time Steps (Last Episode: UAV + BS)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'System_Data_Rate_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 新增：绘制每轮次的reward对比图（UAV拆分为3条曲线）
    plt.figure(figsize=(12, 8))
    x_axis = range(1, Episodes_number+1)
    for uav_id in range(NumberOfUAVs):
        plt.plot(x_axis, Individual_UAV_Reward_seq[uav_id, :], linewidth=2, label=f'UAV{uav_id} Reward', marker='o', markersize=3)
    plt.plot(x_axis, BS_Reward_seq, 'g-', linewidth=2, label='BS Reward', marker='^', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.title('UAVs (split) vs BS Reward per Episode', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'UAV_vs_BS_Reward_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 新增：绘制每回合最后一步的datarate对比图（UAV拆分为3条曲线）
    plt.figure(figsize=(12, 8))
    for uav_id in range(NumberOfUAVs):
        plt.plot(x_axis, Individual_UAV_Final_Datarate_seq[uav_id, :], linewidth=2, label=f'UAV{uav_id} Final Step DataRate', marker='o', markersize=3)
    plt.plot(x_axis, BS_Final_Datarate_seq, 'g-', linewidth=2, label='BS Final Step DataRate', marker='^', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Final Step Data Rate', fontsize=14)
    plt.title('UAVs (split) vs BS Final Step Data Rate per Episode', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'UAV_vs_BS_Final_Datarate_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 新增：绘制每个UAV的reward趋势图
    plt.figure(figsize=(12, 8))
    for uav_id in range(NumberOfUAVs):
        plt.plot(x_axis, Individual_UAV_Reward_seq[uav_id, :], 
                linewidth=2, label=f'UAV{uav_id}', marker='o', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.title('Individual UAV Reward per Episode', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Individual_UAV_Reward_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 新增：绘制每个UAV的最后一步datarate趋势图
    plt.figure(figsize=(12, 8))
    for uav_id in range(NumberOfUAVs):
        plt.plot(x_axis, Individual_UAV_Final_Datarate_seq[uav_id, :], 
                linewidth=2, label=f'UAV{uav_id}', marker='o', markersize=3)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Final Step Data Rate', fontsize=14)
    plt.title('Individual UAV Final Step Data Rate per Episode', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Individual_UAV_Final_Datarate_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 重写：Individual_User_AvgDataRate_NOMA - 每个episode中每个用户的平均数据率
    plt.figure(figsize=(15, 10))
    x_axis = range(1, Episodes_number+1)
    
    # 定义颜色和标记
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    
    # 绘制每个用户的曲线（包含UAV和BS用户）
    for user_id in range(TotalUsers):
        color = colors[user_id % len(colors)]
        marker = markers[user_id % len(markers)]
        plt.plot(x_axis, User_AvgDataRate_seq[user_id, :], 
                linewidth=1.5, label=f'User{user_id}', 
                color=color, marker=marker, markersize=3, alpha=0.8)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Data Rate per Episode', fontsize=14)
    plt.title('Individual User Average Data Rate per Episode (UAV + BS Users)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Individual_User_AvgDataRate_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 重写：UAV_vs_BS_Users_AvgDataRate_NOMA - 每个episode中UAV和BS用户的平均数据率
    plt.figure(figsize=(12, 8))
    x_axis = range(1, Episodes_number+1)
    
    # 计算每个episode中UAV和BS用户的平均数据率
    uav_avg_data_rate_per_episode = []
    bs_avg_data_rate_per_episode = []
    
    for episode in range(Episodes_number):
        # 获取这个episode的用户分配情况
        episode_data = [data for data in env.debug_data['episode_data'] if data['episode'] == episode]
        if episode_data:
            # 使用第一个时间步的用户分配（每个episode开始时分配一次）
            user_association = episode_data[0]['user_association']
            
            # 分离UAV用户和BS用户
            uav_users = [i for i, assoc in enumerate(user_association) if assoc < NumberOfUAVs]
            bs_users = [i for i, assoc in enumerate(user_association) if assoc == NumberOfUAVs]
            
            # 计算这个episode中UAV和BS用户的平均数据率
            if uav_users:
                uav_episode_rates = []
                for data in episode_data:
                    data_rates = data['data_rates']
                    uav_sum_rate = sum(data_rates[i] for i in uav_users)
                    uav_episode_rates.append(uav_sum_rate)
                uav_avg_data_rate_per_episode.append(np.mean(uav_episode_rates))
            else:
                uav_avg_data_rate_per_episode.append(0.0)
            
            if bs_users:
                bs_episode_rates = []
                for data in episode_data:
                    data_rates = data['data_rates']
                    bs_sum_rate = sum(data_rates[i] for i in bs_users)
                    bs_episode_rates.append(bs_sum_rate)
                bs_avg_data_rate_per_episode.append(np.mean(bs_episode_rates))
            else:
                bs_avg_data_rate_per_episode.append(0.0)
        else:
            # 如果没有数据，使用User_AvgDataRate_seq作为备选
            uav_users = [i for i in range(TotalUsers) if i < NumberOfUAVUsers]  # 假设前6个是UAV用户
            bs_users = [i for i in range(TotalUsers) if i >= NumberOfUAVUsers]  # 假设后6个是BS用户
            
            if uav_users:
                uav_avg_data_rate_per_episode.append(np.mean([User_AvgDataRate_seq[user_id, episode] for user_id in uav_users]))
            else:
                uav_avg_data_rate_per_episode.append(0.0)
            
            if bs_users:
                bs_avg_data_rate_per_episode.append(np.mean([User_AvgDataRate_seq[user_id, episode] for user_id in bs_users]))
            else:
                bs_avg_data_rate_per_episode.append(0.0)
    
    # 绘制曲线
    plt.plot(x_axis, uav_avg_data_rate_per_episode, 'r-', linewidth=2, label='UAV Users Average', marker='s', markersize=3)
    plt.plot(x_axis, bs_avg_data_rate_per_episode, 'g-', linewidth=2, label='BS Users Average', marker='^', markersize=3)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Data Rate per Episode', fontsize=14)
    plt.title('UAV Users vs BS Users Average Data Rate per Episode', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'UAV_vs_BS_Users_AvgDataRate_NOMA.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAll results saved in: {result_folder}")
    print(f"GIF animations saved in: {gif_folder}")
    print(f"Output plots saved in: {output_folder}")

if __name__ == '__main__':
    # 固定频段：UAV=3.5GHz，BS=2.4GHz；取消A/B对比
    UAV_FREQ = 3.5
    BS_FREQ = 2.4
    F_c = UAV_FREQ
    run_once()