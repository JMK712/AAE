import math
import random
import time
import csv
from Paradigm.base import SynParadigm
import numpy as np
import mne
from mne.time_frequency import tfr_morlet


def record_data(data):
    # 按分钟记录数据
    with open(r'../data/' + time.strftime('%Y-%m-%d %H-%M') + '.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


class AAEParadigm(SynParadigm):
    def __init__(self, BCIServer):
        SynParadigm.__init__(self, BCIServer=BCIServer)
        self.previous_g_track_code = None
        self.track_mode = 0
        self.triggerEndTrial = None
        self.triggerStartTrial = None
        self.TrialCoefficientData = []

        self.config = {
            'n_session': 5,
            # AAE-track_mode=0实验, AAE-track_mode=1实验, AAE-track_mode=2(无track changing)实验
            # AAE-track_mode=3(random track)实验,对照实验(mute-track_mode=4)
            'n_run': 3,  # 重复实验次数
            'n_trial': 4,  # 道路数量
            'DataPeriod': 1,  # 分析数据的采样周期
            'Road1': False,
            'Road2': False,
            'Road3': False,
            'Road4': False,
            'Pause': False,
        }
        self.running_param = {
            'i_session': -1,  # 参与者数量
            'i_run': -1,  # 重复实验次数
            'i_trial': -1,  # 道路数量
            'is_Listening': False,
            'stream_id': -1,
            'client_id': -1,
        }
        self.ConfigWindow = None

    def run(self):
        self.reset()
        self.SessionLogic()

    def reset(self):
        self.running_param['i_session'] = 0
        self.running_param['i_run'] = 0
        self.running_param['i_trial'] = 0

    def stop(self):
        pass

    async def SessionLogic(self):
        while self.running_param['i_session'] < self.config['n_session']:
            self.running_param['i_session'] += 1
            self.track_mode += 1
            await self.RunLogic()

    async def RunLogic(self):
        while self.running_param['i_run'] < self.config['n_run']:
            self.running_param['i_run'] += 1
            await self.TrialLogic()

    async def TrialLogic(self):
        while self.running_param['i_trial'] < self.config['n_trial']:
            await self.StartTrial()
            self.running_param['i_trial'] += 1
            await self.EndTrial()

    async def StartTrial(self):
        while not self.triggerStartTrial:
            time.sleep(1)  # 以1秒为间隔查看触发bool，等待触发
            self.SendVolume()  # 顺便每一秒发送音量
        self.SendTrackCode(self.track_mode)
        print('Trial ', self.running_param['i_trial'], ' Start')
        self.triggerStartTrial = False

    async def EndTrial(self):
        while not self.triggerEndTrial:
            time.sleep(1)
            self.SendVolume()
        print('Trial ', self.running_param['i_trial'], ' End')
        self.triggerEndTrial = False

    def EventHandler(self, msg_split):

        # ：set receivers :
        # "StartNewTrial"
        # "TrialEnd"
        # "TrialCmd_Report_" + coefficient

        # : set senders
        # "TrialCmd_SetMusicVolume_0.5" ==> o.o - 1.o
        # "StartTrial_1" =>Road 1 . to new a trial by 1234
        # "TrialCmd_SetTrack_0"

        if msg_split[0] == 'StartNewTrial':
            self.triggerStartTrial = True
            self.BCIServer.broadcastCmd('StartTrial_' + str(self.running_param['i_trial'] - 1))

        if msg_split[0] == 'TrialEnd':
            self.triggerEndTrial = True

        if msg_split[0] == 'TrialCmd':
            if msg_split[1] == 'Report':
                self.TrialCoefficientData.append(msg_split[2])

                data = "Session: ", self.running_param['i_session'], "Run: ", self.running_param['i_run'], "Trial", \
                    self.running_param['i_trial'], "Coefficient : " + msg_split[2]
                record_data(data)

                print(data)

    # def compute_attention_scores(self, epochs, freqs=[6, 10]):
    #     """
    #     计算注意力分数。
    #
    #     :param epochs: mne.Epochs 对象
    #     :param freqs: 列表，包含θ波和α波的中心频率
    #     :return: 注意力分数
    #     """
    #     # TODO
    #     n_cycles = freqs / 2.  # 不同频率下的不同周期数
    #
    #     # 计算时间-频率表示
    #     power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
    #                        return_itc=False, decim=3, n_jobs='cuda')
    #
    #     # 提取θ波和α波的功率
    #     theta_power = power[freqs == 6].mean()
    #     alpha_power = power[freqs == 10].mean()
    #
    #     # 计算注意力分数
    #     attention_scores = (alpha_power / theta_power) * 100
    #
    #     return attention_scores
    #
    # def process_data(self, stream_id, data_period=1):
    #     # 获取数据
    #     data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=data_period)
    #
    #     # 创建 Epochs 对象
    #     info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=100, ch_types='eeg')  # 假设采样率为100Hz
    #     epochs = mne.EpochsArray(data, info, tmin=0)
    #
    #     # 计算注意力分数
    #     attention_scores = self.compute_attention_scores(epochs)
    #     return attention_scores

    def SendVolume(self):
        # 生成并发送音量数据
        print('generating Volume.')
        stream_id = self.running_param['stream_id']
        data_period = self.config['DataPeriod']
        client_id = self.running_param['client_id']

        # 使用mne来获取注意力状态，0.0-1.0
        # TODO determine the explicit attention_scores
        # volume = self.process_data(stream_id=stream_id, data_period=data_period)  # 选定通道的α和θ波平均功率折合音量
        volume = self.calculate_attention(self.running_param['stream_id'], self.config['DataPeriod'])  # 选定通道的平均功率折合音量
        self.BCIServer.broadcastCmd("TrialCmd_SetMusicVolume_" + volume)
        print('Sent Volume:', volume)

    def SendTrackCode(self, track_mode=0):
        g_code = self.GenerateTrackCode()
        if track_mode == 0:
            print('generating TrackCode by average Coefficient.')
            track_code = g_code
        elif track_mode == 1:
            print('generating TrackCode by gradually changing method.')
            if self.previous_g_track_code > g_code:
                track_code = self.previous_g_track_code + 1
            elif self.previous_g_track_code < g_code:
                track_code = self.previous_g_track_code - 1
            elif self.previous_g_track_code == g_code:
                track_code = g_code
        elif track_mode == 2:
            print('generating TrackCode by no change mode')
            track_code = self.previous_g_track_code
        elif track_mode == 3:
            print('generating TrackCode by random mode')
            track_code = random.randint(0, 10)
        elif track_mode == 4:
            print('generating TrackCode by no track (mute) mode')
            track_code = -1
        self.BCIServer.broadcastCmd("TrialCmd_SetTrack_" + str(track_code))
        print('Sent TrackCode:', track_code)
        self.previous_g_track_code = track_code

    def GenerateTrackCode(self):
        # TODO check if valid
        track_code = '0'
        for i in range(self.TrialCoefficientData.count()):
            track_code += float(self.TrialCoefficientData[i])
        track_code = abs(math.ceil(int(track_code) / self.TrialCoefficientData.count()) * 10)
        return track_code

    def calculate_attention(self, attention_channels, window_size=1):

        client = self.BCIServer.streamClients[self.running_param['stream_id']]

        if not client.buffer:
            return None

        data = client.Buffer.getData(
            window_size * client.basicInfo['sampleRate'])  # Get the latest data from the buffer

        if data is None or data.shape[1] == 0:
            return None

        relevant_data = data[attention_channels, :]  # Extract the relevant channels
        mean_powers = np.mean(np.square(relevant_data), axis=1)  # Calculate the mean power for each channel
        # Normalize the mean powers to a 0.0-1.0 scale
        max_power = np.max(mean_powers)
        min_power = np.min(mean_powers)
        normalized_powers = (mean_powers - min_power) / (max_power - min_power)
        attention_value = np.mean(normalized_powers)  # Average the normalized powers to get a single attention value
        return attention_value
