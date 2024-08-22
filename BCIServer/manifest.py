import mne

BCIC_dir = 'D:\\BCI\\BCICdatasets'
mne_path = 'D:\\BCI\\mne_data'

'''
mne.set_config('MNE_DATA',mne_path)
mne.set_config('MNE_DATASETS_ALEXEEG_PATH',mne_path)
mne.set_config('MNE_DATASETS_BNCI_PATH',mne_path)
mne.set_config('MNE_DATASETS_BRAININVADERS_PATH',mne_path)
mne.set_config('MNE_DATASETS_EEGBCI_PATH',mne_path)
mne.set_config('MNE_DATASETS_GIGADB_PATH',mne_path)
mne.set_config('MNE_DATASETS_MAMEM1_PATH',mne_path)
mne.set_config('MNE_DATASETS_MAMEM2_PATH',mne_path)
mne.set_config('MNE_DATASETS_MAMEM3_PATH',mne_path)
mne.set_config('MNE_DATASETS_NAKANISHI_PATH',mne_path)
mne.set_config('MNE_DATASETS_SSVEPEXO_PATH',mne_path)
mne.set_config('MNE_DATASETS_WANG_PATH',mne_path)


#For Unity BCIPlugin
MItype = {
    0:'LeftHand',
    1:'RightHand',
    2:'Tongue',
    3:'LeftFoot',
    4:'RightFoot',
    5:'Rest'
}

'''
# 没有找到mne的set_config的keys列表，于是解决不了mne非标准key值报错，运行一遍写入json后权当是其他部分会调用吧。。
