import asyncio
import math
import random
import time
import csv
from Paradigm.base import SynParadigm
import numpy as np
import mne


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

        self.sfreq = 1000  # frequency
        # self.ch_names = ['Fp1', 'Fp2', 'Fpz', 'C3', 'C4', 'Cz', 'FC3', 'FC4', 'FCz', 'P3', 'P4', 'Pz', 'T7', 'T8']
        self.ch_names = ['1', '2', "3", '4', "5", '6', '7', '8']
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')

        self.config = {
            'n_session': 5,
            # AAE-track_mode=0实验, AAE-track_mode=1实验, AAE-track_mode=2(无track changing)实验
            # AAE-track_mode=3(random track)实验,对照实验(mute-track_mode=4)
            'n_run': 3,  # 重复实验次数
            'n_trial': 4,  # 道路数量
            'DataPeriod': 1,  # 分析数据的采样周期
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
        self.BCIServer.eventService.typeChangedHandler.update({'AAE': self.EventHandler})

        asyncio.run(self.async_main())

    async def async_main(self):
        self.triggerStartTrial = asyncio.Event()
        self.triggerEndTrial = asyncio.Event()
        await asyncio.gather(
            self.SessionLogic(),
            self.SendVolumeLoop()
        )

    def reset(self):
        self.running_param['i_session'] = 0
        self.running_param['i_run'] = 0
        self.running_param['i_trial'] = 0

    def stop(self):
        pass

    async def SessionLogic(self):
        while self.running_param['i_session'] < self.config['n_session']:
            self.running_param['i_session'] += 1
            await self.RunLogic()
            self.track_mode += 1

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
        await self.triggerStartTrial.wait()
        self.triggerStartTrial.clear()
        self.SendTrackCode(self.track_mode)
        print('Trial ', self.running_param['i_trial'] + 1, ' Start')

    async def EndTrial(self):
        await self.triggerEndTrial.wait()
        self.triggerEndTrial.clear()
        print('Trial ', self.running_param['i_trial'] + 1, ' End')

    def EventHandler(self, msg_split):

        # ：set receivers :
        # "StartNewTrial"
        # "TrialEnd"
        # "TrialCmd_Report_" + coefficient

        # : set senders
        # "TrialCmd_SetMusicVolume_0.5" ==> o.o - 1.o
        # "StartTrial_1" =>Road 1 . to new a trial by 1234
        # "TrialCmd_SetTrack_0"
        print(msg_split)
        if msg_split[0] == 'StartNewTrial':
            self.triggerStartTrial.set()
            self.BCIServer.broadcastCmd('StartTrial_' + str(self.running_param['i_trial']))
            print('StartTrial_' + str(self.running_param['i_trial']))

        if msg_split[0] == 'TrialEnd':
            self.triggerEndTrial.set()

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

    async def SendVolumeLoop(self):
        while True:
            # 生成并发送音量数据
            print('generating Volume.')
            # 使用mne进行滤波，生成注意力状态，0.0-1.0
            # determine the explicit attention_scores
            # volume = self.process_data(stream_id=stream_id, data_period=data_period)  # 选定通道的α和θ波平均功率折合音量
            volume = self.calculate_attention(self.ch_names, self.config['DataPeriod'])  # 选定通道的平均功率折合音量
            if volume is not None:
                self.BCIServer.broadcastCmd("TrialCmd_SetMusicVolume_" + str(volume))
            else:
                self.BCIServer.broadcastCmd("TrialCmd_SetMusicVolume_" + str(0.05))
            self.TrialCoefficientData.append(volume)
            print('Sent Volume:', volume)
            await asyncio.sleep(1)

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
        self.TrialCoefficientData = []
        self.previous_g_track_code = track_code

    def GenerateTrackCode(self):
        # TODO check if valid
        track_code = '0'

        try:
            for i in range(self.TrialCoefficientData.count()):
                track_code += float(self.TrialCoefficientData[i])
            track_code = abs(math.ceil(int(track_code) / self.TrialCoefficientData.count()) * 10)
        except:
            track_code = str(random.randint(1, 10))
        return track_code

    def calculate_attention(self, attention_channels, window_size=1):
        client = self.BCIServer.streamClients[self.running_param['stream_id']]
        data = client.Buffer.getData(window_size)  # Get the latest data from the buffer

        if data is None or data.shape[1] == 0:
            print("No data available in buffer.")
            return None

        # Ensure the order of attention_channels matches the order in data
        # Check if all channels in attention_channels are present in data
        channel_indices = []
        for channel in attention_channels:
            if channel in self.ch_names:
                channel_indices.append(self.ch_names.index(channel))
            else:
                print(f"Warning: Channel {channel} not found in data. Skipping.")

        if len(channel_indices) == 0:
            print("No valid channels found in data.")
            return None

        # If there's only one channel in data, use it directly
        if data.shape[0] == 1:
            relevant_data = data
        else:
            # Check if the number of channels in data matches the number of channels requested
            if data.shape[0] < len(channel_indices):
                print(
                    f"Warning: Not enough channels in data ({data.shape[0]}). Expected {len(channel_indices)} channels.")
                return None

            relevant_data = data[channel_indices, :]  # Extract the relevant channels by their indices

        raw = mne.io.RawArray(relevant_data, self.info)

        # Apply band-pass filter
        raw.filter(8, 12, method='iir')  # Filter in the alpha band (8-12 Hz)

        # Extract features, e.g., average power in the alpha band
        # Set n_per_seg to allow zero-padding
        n_per_seg = data.shape[1]  # Use the actual signal length
        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=8, fmax=12, n_fft=256, n_per_seg=n_per_seg)
        mean_alpha_power = np.mean(psds, axis=1)

        # Normalize the mean powers to a 0.0-1.0 scale
        max_power = np.max(mean_alpha_power)
        min_power = np.min(mean_alpha_power)
        normalized_powers = (mean_alpha_power - min_power) / (max_power - min_power)
        attention_value = np.mean(normalized_powers)  # Average the normalized powers to get a single attention value
        return attention_value
