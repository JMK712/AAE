"""
创建脑电实验的步骤
1. 在python端写一个Paradigm类
2. 在Unity端写一个对应的Paradigm类
3. 打开BCIServer连接，开始
"""

import time
import torch

import numpy as np
from threading import Thread,Timer
from copy import deepcopy

from PluginCore.Processor.base import preprocess
from BCIServer.base import BCIServer
from StreamClient.Curry.base import CurryClient

class KKMIParadigm():
    def __init__(self, BCIServer, preprocessor, trainned_model, client_id,
                 dataPeriod, stream_id, update_interval):
        self.BCIServer = BCIServer
        self.preprocessor = preprocessor
        self.trainned_model = trainned_model
        self.dataPeriod = dataPeriod
        self.update_interval = update_interval

        info = self.BCIServer.streamClients[stream_id].infoList['mne_info']
        info = info.pick_channels(info.ch_names[0:64])
        self.eeg_info = info

        self.stream_id = stream_id
        self.client_id = client_id
        self.is_running = False

    def run(self):
        self.is_running = True

        self.running_thread = Thread(target=self.run_loop)
        self.running_thread.start()

    def stop(self):
        self.is_running = False

    def run_loop(self):
        while self.is_running:
            init_time = time.time()

            data_orig = self.BCIServer.streamClients[self.stream_id].Buffer.getData(dataPeriod=self.dataPeriod)[0:64, :]
            if self.preprocessor is not None:
                data = preprocess(data_orig, self.preprocessor, apply_on_array=True, info=self.eeg_info)
            else:
                data = deepcopy(data_orig)

            #data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            try:
                state = self.trainned_model.predict(data)[0]
            except:
                data = torch.Tensor(data).cuda()
                state = self.trainned_model.predict(data)[0]
            # y = self.trainned_model.predict(data)[0]
            # state = int(y)

            self.BCIServer.valueService.SetValue('MIstate', state)
            self.BCIServer.valueService.UpdateValue(name='MIstate', value=state, conn=self.BCIServer.appClients[self.client_id])

            end_time = time.time()

            decode_time = end_time - init_time

            if decode_time<self.update_interval:
                time.sleep(float(self.update_interval-decode_time))


stream_client = CurryClient()
bs = BCIServer()


from Experiments.chapter4.utils import *
#__________________________________________Editable____________________________________
i_subject = 0
buffer_length = 8
trial_length = 4

model_name = 'ShallowConvNet'
model_idx = 4
model_exp_date = '23-9-10'
model_data_source = CALIBRATE_DIR_NAME
model_info = load_model_from_pkl(model_name=model_name, subject_id=i_subject, model_idx=model_idx, exp_date=model_exp_date
                                 , data_source=model_data_source)
trained_model = model_info['model']
preprocessor = model_info['preprocess']
update_interval = 1
#__________________________________________Editable____________________________________
server = BCIServer()
server.run()

sc = CurryClient(BufferPeriod=buffer_length)
sc.startStreaming()
server.loadStreamClient(sc)
i_streamClient = list(server.streamClients.keys())[0]
server.listenUnityAppClient()














#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
i_appClient = list(server.appClients.keys())[0]

server.loadParadigm(KKMIParadigm(BCIServer=server, preprocessor=preprocessor, trainned_model=trained_model,
                                 dataPeriod=trial_length, stream_id=i_streamClient, update_interval=update_interval
                                ,client_id=i_appClient))
i_paradigm = list(server.paradigms.keys())[0]


server.paradigms[i_paradigm].run()
