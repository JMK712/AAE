import os
import time
import csv


def record_data(data):
    # 按分钟记录数据
    timestamp = time.strftime('%Y-%m-%d %H-%M')
    filename = f'./data/{timestamp}.csv'

    # 确保目录存在
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 处理数据，确保每个元素都是一个单独的值
    processed_data = [item.strip() for item in data.split(',')]

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(processed_data)
