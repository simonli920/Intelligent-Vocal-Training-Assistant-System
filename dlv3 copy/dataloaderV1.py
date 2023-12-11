import os
from torch.utils.data import DataLoader,Dataset
import re
import random
from preprocess import middle_samples, specgrams
import librosa

class MyDataset_v0(Dataset):            # 继承Dataset
    def __init__(self, path_list, path2num):     # __init__是初始化该类的一些基础参数
        self.sample_list = []
        label_map = {'0': None, '1': 0, '2': None, '3': 1, '4': 2}
        for filepath in path_list:
            label = label_map[filepath[-5]]
            if not(filepath[-3:] =='wav' and label and os.path.exists(filepath)):
                continue
            voice_array, sample_rate = librosa.load(filepath, sr=48000)
            # for voice_segment in window_cut(voice_array, sample_rate, lens_win=200, step=100):
            for voice_segment in middle_samples(voice_array, sample_rate, lens=200, pad_type='none'):
                specgram = specgrams['mel'](voice_segment, sample_rate, 112, 100)
                self.sample_list.append({'voice_data': voice_segment, 'specgram': specgram,
                                         'label': label, 'path_num': path2num[filepath]})
        random.shuffle(self.sample_list)

    def __len__(self):            # 返回整个数据集的大小
        return len(self.sample_list)

    def __getitem__(self, index):          # 根据索引index返回dataset[index]
        return self.sample_list[index]     # 返回该样本


# 路径解析为文件信息
def path2fileinf(path):
    pattern = re.compile(r'.*\\(.*)\\(.*)\\(.*)\\(.*)\.wav$')
    result = re.findall(pattern, path)
    if not result:
        return None
    people, device, scene, filename = result[0]
    scene = re.findall(r'bubble|vowellong|vowelshort|fade|staccato', scene)[0]
    label = filename[-1]
    son_scene = re.findall(r'_(\D.?)_\d*_\d$', filename)
    son_scene = son_scene[0] if son_scene else 'None'
    inf_dict = {'people': people, 'device': device, 'scene': scene,
                'son_scene': son_scene, 'filename': filename, 'label': label}
    return inf_dict
