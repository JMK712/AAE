import asyncio
import csv
import math
import os
import random
import time

import mne
import numpy as np

from Paradigm.base import SynParadigm


def record_data(data, need_process=True):
    # 按分钟记录数据
    timestamp = time.strftime('%Y-%m-%d_%H-%M')  # 使用下划线替换空格
    filename = f'./data/{timestamp}.csv'

    # 确保目录存在
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if need_process:
        # 将元组中的字符串合并成一个单独的字符串
        data_str = ''.join(str(data))

        # 处理数据，确保每个元素都是一个单独的值
        processed_data = [item.strip() for item in data_str.split(',')]
    else:
        processed_data = data

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)  # 设置 quotechar 和 quoting 参数
        writer.writerow(processed_data)


class AAEParadigm(SynParadigm):
    def __init__(self, BCIServer, stream_id):
        SynParadigm.__init__(self, BCIServer=BCIServer)
        self.previous_g_track_code = None
        self.track_mode = -1
        self.previous_g_track_code = 0
        self.triggerEndTrial = None
        self.triggerStartTrial = None
        self.max_power = 50
        self.min_power = 0
        self.TrialCoefficientData = []
        self.TrialAttentionData = []

        self.stream_id = stream_id

        self.sfreq = 1000  # frequency
        # self.ch_names = ['Fp1', 'Fp2', 'Fpz', 'C3', 'C4', 'Cz', 'FC3', 'FC4', 'FCz', 'P3', 'P4', 'Pz', 'T7', 'T8']
        self.ch_names = ['1', '2', "3", '4', "5", '6', '7', '8']
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')

        self.config = {
            'n_session': 5,
            # AAE-track_mode=-1实验，作为Demo实验进行一遍四种场景，获取标准化注意力所需的功率最大最小值数据
            # AAE-track_mode=0实验, AAE-track_mode=1实验, AAE-track_mode=2(无track changing)实验
            # AAE-track_mode=3(random track)实验,对照实验(mute-track_mode=4)
            'n_run': 5,  # 重复实验次数
            'n_trial': 4,  # 道路数量(0也算)
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
        self.BCIServer.eventService.typeChangedHandler = {}
        exit()
        pass

    async def SessionLogic(self):
        while self.running_param['i_session'] < self.config['n_session']:
            self.running_param['i_session'] += 1
            data = "track_mode : " + str(self.track_mode)
            record_data(data, need_process=True)
            if self.track_mode == -1:
                # 进入演示模式
                await self.DemoLogic()
            else:
                await self.RunLogic()
            self.track_mode += 1
            self.running_param['i_run'] = 0

        if self.running_param["i_session"] == self.config["n_session"]:
            print("session now : "+ str(self.running_param["i_session"]))
            print("n_session config : " + str(self.config["n_session"]))
            print("**************************** auto stop activated ************************************")
            self.stop()

    async def DemoLogic(self):
        """当 track_mode == -1 时，执行演示逻辑"""
        print("entered demologic")
        modified_n_run = self.config['n_run']
        self.config['n_run'] = 1
        print("set n_run = "+str(self.config['n_run']))

        for mode in range(4):  # 四种模式的演示,第五种就不demo了因为没有意义。
            self.track_mode = mode
            print("demolocic _ mode : ", mode)
            await self.RunLogic()
            self.running_param['i_run'] = 0
            print("demologic_generating max data")

            print(self.TrialAttentionData)
            flattened_data = [item for sublist in self.TrialAttentionData for item in sublist]
            print(flattened_data)
            current_max = max(flattened_data)
            current_min = min(flattened_data)
            print(f"Max Power: {current_max}, Min Power: {current_min}")
            if self.max_power < current_max < 9999:
                self.max_power = current_max
            if self.min_power == 0 or current_min < self.min_power:
                self.min_power = current_min
            print("max_power : ", self.max_power)
            print("min_power : ", self.min_power)
            self.TrialAttentionData = []  # 清空数据以便下次试验
        # 记录最大值和最小值
        record_data(f"Max Power: {self.max_power}, Min Power: {self.min_power}", need_process=True)
        self.config['n_run'] = modified_n_run
        print("change n_run back to : "+str(self.config['n_run']))
        print("out from demologic")

    async def RunLogic(self):
        while self.running_param['i_run'] < self.config['n_run']:
            self.running_param['i_run'] += 1
            await self.TrialLogic()
            self.running_param['i_trial'] = 0

    async def TrialLogic(self):
        while self.running_param['i_trial'] < self.config['n_trial']:
            await self.StartTrial()
            await self.EndTrial()
            self.running_param['i_trial'] += 1

    async def StartTrial(self):
        await self.triggerStartTrial.wait()
        self.triggerStartTrial.clear()
        self.SendTrackCode(self.track_mode)
        record_data("Sent Track mode "+str(self.track_mode), need_process=True)
        print('Trial ', self.running_param['i_trial'], ' Start')

    async def EndTrial(self):
        await self.triggerEndTrial.wait()
        self.triggerEndTrial.clear()
        print('Trial ', self.running_param['i_trial'], ' End')

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
                # self.TrialCoefficientData.append(msg_split[2])

                data = "Session: " + str(self.running_param['i_session'])+"; Run: "+str(self.running_param['i_run']) +\
                       "; Trial" + str(self.running_param['i_trial'])+"; Coefficient : " + str(msg_split[2])
                record_data(data, need_process=True)

                print(data)

    async def SendVolumeLoop(self):
        while True:

            print('generating Volume.')

            volume = self.calculate_attention(self.ch_names, self.config['DataPeriod'], 1)  # 选定通道的平均功率折合音量
            if volume is not None:
                self.BCIServer.broadcastCmd("TrialCmd_SetMusicVolume_" + str(volume))
            else:
                self.BCIServer.broadcastCmd("TrialCmd_SetMusicVolume_" + str(0.05))
            self.TrialCoefficientData.append(volume)

            attention = self.calculate_attention(self.ch_names, self.config['DataPeriod'], 0)  # 选定通道的平均功率折合音量
            self.TrialAttentionData.append(attention)

            data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + " volume : " + str(volume)
            record_data(data, need_process=True)
            print('Sent Volume:', volume)
            await asyncio.sleep(1)

    def SendTrackCode(self, track_mode=-1):
        g_code = int(self.GenerateTrackCode())
        track_code = 1
        if track_mode == -1:
            track_code = -1
            print("Demo period start")
        elif track_mode == 0:
            print('generating TrackCode by average Coefficient.')
            track_code = g_code
        elif track_mode == 1:
            print('generating TrackCode by gradually changing method.')
            if self.previous_g_track_code > g_code and self.previous_g_track_code != 10:
                track_code = self.previous_g_track_code + 1
            elif self.previous_g_track_code < g_code and self.previous_g_track_code != 0:
                track_code = self.previous_g_track_code - 1
            else:
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
        record_data("Sent Track code: "+str(track_code), need_process=True)
        self.TrialCoefficientData = []
        self.previous_g_track_code = int(track_code)

    def GenerateTrackCode(self):
        #  check if valid
        track_code = 0

        try:
            for i in range(self.TrialCoefficientData.count()):
                track_code += float(self.TrialCoefficientData[i])
            track_code = abs(math.ceil(int(track_code) / self.TrialCoefficientData.count()) * 10)
        except:
            track_code = str(random.randint(1, 10))
        return track_code

    def calculate_attention(self, attention_channels, window_size=1, volume_mode=1):
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
                    f"Warning: Not enough channels in data ({data.shape[0]}).Expected {len(channel_indices)} channels.")
                return None

            relevant_data = data[channel_indices, :]  # Extract the relevant channels by their indices

        raw = mne.io.RawArray(relevant_data, self.info)

        # Apply band-pass filter
        raw.filter(8, 12, method='iir')  # Filter in the alpha band (8-12 Hz)

        # Extract features, e.g., average power in the alpha band
        # Set n_per_seg to allow zero-padding
        n_per_seg = data.shape[1]  # Use the actual signal length
        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=8, fmax=12, n_fft=1000, n_per_seg=n_per_seg)
        mean_alpha_power = np.mean(psds, axis=1)

        max_power = np.max(mean_alpha_power)
        min_power = np.min(mean_alpha_power)

        print("*****************", max_power)
        print("*****************", self.max_power)

        if volume_mode == 0:
            return max_power, min_power
        # Normalize the mean powers to a 0.0-1.0 scale
        # made a demo precess to estimate the fixed max power

        normalized_powers = (np.mean(mean_alpha_power) - self.min_power) / (self.max_power - self.min_power)
        print("*****************", normalized_powers)
        attention_value = np.mean(normalized_powers)  # Average the normalized powers to get a single attention value
        if attention_value > 1:
            print("attention value too big:", attention_value, ", eliminated.")
            return 0
        return attention_value
