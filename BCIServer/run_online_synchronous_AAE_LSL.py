from Paradigm.AAEParadigm import AAEParadigm
from StreamClient.LSL.base import LSLClient
from BCIServer.base import BCIServer
# import preprocess_test


# 启动BCIPlugin服务（使用默认核心）
server = BCIServer(pluginCore=None)
server.run()

# 启动Unity监听客户端
server.listenUnityAppClient()
i_appClient = list(server.appClients.keys())[0]

# TODO：设置需要的通道，并相应的启动LSL流
ch_names = ['T7', 'T8', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
sc = LSLClient(ch_names=ch_names)
sc.startStreaming()

# todo：将LSL流加载到BCIPlugin服务中
server.loadStreamClient(sc)

# 加载范式
server.loadParadigm(AAEParadigm(BCIServer=server, preprocess=None))

i_paradigm = list(server.paradigms.keys())[0]
i_streamClient = list(server.streamClients.keys())[0]

server.paradigms[i_paradigm].running_param['stream_id'] = i_streamClient
server.paradigms[i_paradigm].running_param['client_id'] = i_appClient

# 调整config
server.paradigms[i_paradigm].config['n_session'] = 1  # 总session数->总参与者数
server.paradigms[i_paradigm].config['n_run'] = 3  # 一个Session下的run数->每人的实验次数
server.paradigms[i_paradigm].config['n_trial'] = 4  # 一次实验下的阶段数->四条道路
server.paradigms[i_paradigm].config['DataPeriod'] = 4  # TODO：run_loop()函数中执行的就是取前DataPeriod秒的数据，使用分类器分类。之后调用运动想象状态同步服务同步到Unity。
server.paradigms[i_paradigm].config['Road1'] = True
server.paradigms[i_paradigm].config['Road2'] = True
server.paradigms[i_paradigm].config['Road3'] = True
server.paradigms[i_paradigm].config['Road4'] = True
server.paradigms[i_paradigm].config['Pause'] = True

# 开始运行范式
server.paradigms[i_paradigm].run()
# 结束运行范式
server.paradigms[i_paradigm].stop()
