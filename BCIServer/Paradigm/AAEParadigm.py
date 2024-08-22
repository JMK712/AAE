"""
为方便起见，不对Unity的ValueService进行重写，仅在AAEOnlineParadigm.cs中做了映射：
LeftHand -> Road 1
RightHand -> Road 2
Tongue,-> Road 3
LeftFoot -> Road 4
RightFoot -> Pause
Rest
标注以方便开发，传输信号时需要逆映射回去
"""
import time
import numpy as np
import csv
from Paradigm.base import SynParadigm
import mne
from mne.time_frequency import tfr_morlet


class AAEParadigm(SynParadigm):
    def __init__(self, BCIServer, preprocess, trainned_model, inverse_class_map=None,
                 config=None, log_func=print):
        SynParadigm.__init__(self, BCIServer=BCIServer)
        self.init_config(config)
        self.running_param = {
            'i_session': -1,  # 参与者数量
            'i_run': -1,  # 重复实验次数
            'i_trial': -1,  # 道路数量
            'is_Listening': False,
            'stream_id': -1,
            'client_id': -1,
        }
        self.log_func = log_func
        self.ConfigWindow = None

        self.preprocess = preprocess
        self.trainned_model = trainned_model
        self.inverse_class_map = inverse_class_map

    def configParadigm(self):
        pass

    def SendAAETrial(self, name):
        option = 'On' if self.config[name] else 'Off'
        cmd = 'Cmd_AAEOnline_Set' + name + option
        self.BCIServer.broadcastCmd(cmd)

    # def SendConfig(self):
    #     delay = 1
    #     pbar = tqdm(total=10)
    #
    #     self.BCIServer.broadcastCmd('Cmd_AAEOnline_SetNSession_' + str(self.config['n_session']))
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.BCIServer.broadcastCmd('Cmd_AAEOnline_SetNRun_' + str(self.config['n_run']))
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.BCIServer.broadcastCmd('Cmd_AAEOnline_SetNTrial_' + str(self.config['n_trial']))
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.SendAAETrial('Road1')
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.SendAAETrial('Road2')
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.SendAAETrial('Road3')
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.SendAAETrial('Road4')
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     self.SendAAETrial('Pause')
    #     time.sleep(delay)
    #     pbar.update(1)
    #
    #     time.sleep(delay)
    #     pbar.close()

    def init_config(self, config):
        # 默认是无限进行，部位无设置默认为否
        default_config = {
            'n_session': 1,
            'n_run': 3,
            'n_trial': 4,
            'DataPeriod': -1,
            'Road1': False,
            'Road2': False,
            'Road3': False,
            'Road4': False,
            'Pause': False,
        }
        if config is None:
            self.config = default_config
        else:
            for k in default_config:
                self.config[k] = config[k]

    def run(self):
        self.reset()
        self.startListening()
        self.BCIServer.broadcastCmd('Cmd_AAEOnline_Start')

    def reset(self):
        self.running_param['i_session'] = 0
        self.running_param['i_run'] = 0
        self.running_param['i_trial'] = 0

    def stop(self):
        self.stopListening()

    def EventHandler(self, type):

        #TODO：set receivers :
        # "StartTrial_1" =>Road 1
        # "TrialEnd"
        # "TrialCmd_Report_" + coefficient
        # "TrialCmd_RequestTrack_" + track
        # "StartNewTrial"
        #TODO: set senders
        # "TrialCmd_SetMusicVolume_0.5" ==> o.o - 1.o

        # if type == 'session start':
        #     self.running_param['i_session'] += 1
        #     self.running_param['i_run'] = -1
        #     self.running_param['i_trial'] = -1
        #     print('Session ', self.running_param['i_session'] + 1, ' Start')
        #
        # if type == 'session end':
        #     print('Session ', self.running_param['i_session'] + 1, ' End')
        #
        # if type == 'run start':
        #     self.running_param['i_run'] += 1
        #     self.running_param['i_trial'] = -1
        #     print('Run ', self.running_param['i_run'] + 1, ' Start')
        #
        # if type == 'run end':
        #     print('Run ', self.running_param['i_run'] + 1, ' End')

        if type == 'trial start':
            self.running_param['i_trial'] += 1
            print('Trial ', self.running_param['i_trial'] + 1, ' Start')

        if type == 'trial end':
            print('Trial ', self.running_param['i_trial'] + 1, ' End')

        if type == 'request volume data':
            print('MI Online Logic: Request received, generating Cmd...')

            stream_id = self.running_param['stream_id']
            dataPeriod = self.config['DataPeriod']
            client_id = self.running_param['client_id']
            data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)
            data = np.expand_dims(data, axis=0)

            # TODO: 使用mne来获取注意力状态，0表示不集中，1表示集中

            self.pro.process_data(stream_id=1, client_id=1)

        if type == 'request track data':
            print('MI Online Logic: Request received, generating Cmd...')

            stream_id = self.running_param['stream_id']
            dataPeriod = self.config['DataPeriod']
            client_id = self.running_param['client_id']
            data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)
            data = np.expand_dims(data, axis=0)

            # TODO: 使用mne来获取注意力状态，0表示不集中，1表示集中
            # 使用模型来预测data
            y = self.trainned_model.predict(data)[0]

    def startListening(self):
        self.BCIServer.eventService.typeChangedHandler.update({'AAE': self.EventHandler})
        self.running_param['is_Listening'] = True

    def stopListening(self):
        if self.running_param['is_Listening']:
            self.BCIServer.eventService.typeChangedHandler.pop('AAE')
            self.running_param['is_Listening'] = False

    def compute_attention_scores(epochs, freqs=[6, 10]):
        """
        计算注意力分数。

        :param epochs: mne.Epochs 对象
        :param freqs: 列表，包含θ波和α波的中心频率
        :return: 注意力分数
        """
        n_cycles = freqs / 2.  # 不同频率下的不同周期数

        # 计算时间-频率表示
        power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                           return_itc=False, decim=3, n_jobs='cuda')

        # 提取θ波和α波的功率
        theta_power = power[freqs == 6].mean()
        alpha_power = power[freqs == 10].mean()

        # 计算注意力分数
        attention_scores = (alpha_power / theta_power) * 100

        return attention_scores

    def process_data(self, stream_id, client_id, dataPeriod=1):
        # 获取数据
        data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)

        # 创建 Epochs 对象
        info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=100, ch_types='eeg')  # 假设采样率为100Hz
        epochs = mne.EpochsArray(data, info, tmin=0)

        # 计算注意力分数
        attention_scores = self.compute_attention_scores(epochs)

        # 更新服务中的注意力值
        self.BCIServer.valueService.SetValue('Attention', attention_scores)
        self.BCIServer.valueService.UpdateValue(name='Attention', value=attention_scores,
                                                conn=self.BCIServer.appClients[client_id])

    def RecordData(self, data):
        # 按分钟记录数据
        with open(r'../data/' + time.strftime('%Y-%m-%d %H-%M') + '.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
