import mne
import numpy as np

# 文件路径
raw_file = r'D:\BCI\curry_data\Acquisition 917.cdt'  # Curry 8 原始数据文件

# 读取原始数据
raw = mne.io.read_raw_curry(raw_file, preload=True)

# 预处理数据
raw.filter(l_freq=0.5, h_freq=40.0)  # 应用带通滤波器

# 选择与注意力相关的通道
# 这里假设 Fz, FCz, Cz 是与注意力相关的通道
attention_channels = ['Fz', 'FCz', 'Cz', 'T8']

# 选择时间窗口
# 这里假设刺激呈现后 200ms 到 500ms 之间是注意力集中的时间段
time_window = (0.2, 20)

# 提取感兴趣的时间段内的数据
data, times = raw[attention_channels, :]
start_time_index = np.where(times >= time_window[0])[0][0]
end_time_index = np.where(times <= time_window[1])[0][-1]

# 计算选定通道和时间窗口内的平均值
avg_attention_data = np.mean(data[:, start_time_index:end_time_index], axis=1)

print("Average attention data across channels:", avg_attention_data)