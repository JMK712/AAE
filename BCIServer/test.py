import numpy as np

from Paradigm.base import SynParadigm
from tqdm import tqdm
import time
import mne
from mne.time_frequency import tfr_morlet

eeg_data = mne.io.read_raw_curry(r'D:\BCI\curry_data\Acquisition 917.cdt')


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
    #data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)
    data = eeg_data.get_data()

    # 创建 Epochs 对象
    info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=100, ch_types='eeg')  # 假设采样率为100Hz
    epochs = mne.EpochsArray(data, info, tmin=0)

    # 计算注意力分数
    attention_scores = self.compute_attention_scores(epochs)
    print(attention_scores)
    # 更新服务中的注意力值 self.BCIServer.valueService.SetValue('Attention', attention_scores)
    # self.BCIServer.valueService.UpdateValue(name='Attention', value=attention_scores,
    # conn=self.BCIServer.appClients[client_id])


process_data(self=None, stream_id=None, client_id=None, dataPeriod=1)
