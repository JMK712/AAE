import os

from BCIServer.base import BCIServer
from StreamClient.Curry.base import CurryClient
from Paradigm.Cursor.base import CursorCalibrateParadigm

from Experiments.chapter4.manifest import *

#__________________________________________Editable____________________________________
exp_date = '23-9-10'    # 23-3-6
i_run = '4'             # '1'
i_subject = 0
save_dir = os.path.join(DATASET_DIR, SUBJECT_IDS[i_subject], CALIBRATE_DIR_NAME, exp_date)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

buffer_length = 8
trial_length = 4
#__________________________________________Editable____________________________________

server = BCIServer()
server.run()

sc = CurryClient(BufferPeriod=buffer_length)
sc.startStreaming()

server.loadStreamClient(sc)
i_streamClient = list(server.streamClients.keys())[0]

server.listenUnityAppClient()

server.loadParadigm(CursorCalibrateParadigm(BCIServer=server,dataPeriod=trial_length,i_streamClient=i_streamClient))
i_paradigm = list(server.paradigms.keys())[0]

server.paradigms[i_paradigm].run()












server.paradigms[i_paradigm].stop()

info = server.streamClients[i_streamClient].infoList['mne_info']
windowedDataset = server.paradigms[i_paradigm].createDataset(trial_start=0,
                                                                 trial_length=trial_length,
                                                                 subject_id=i_subject,
                                                                 info=info)
server.paradigms[i_paradigm].save_calibrate_to_pkl_file(windowedDataset=windowedDataset,
                                                        filepath=save_dir,
                                                        filename=CALIBRATE_DATASET_NAME+i_run)


