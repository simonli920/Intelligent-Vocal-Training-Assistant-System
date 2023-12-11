import os
from torch.utils.data import DataLoader,Dataset
import re
import random
from preprocess import window_cut, middle_cut, specgrams
import librosa


class MyDataset(Dataset):            # 继承Dataset
    def __init__(self, setting):     # __init__是初始化该类的一些基础参数
        self.setting = setting       # 相关设置均转移至setting.py文件里
        self.sample_list = []
        label_map = {'0': None, '1': 0, '2': None, '3': 1, '4': 2}
        label_map = {'1': 0, '3': 1, '4': 2}
        for people in setting.people_list:
            print('正在读取%s的数据' % people)
            people_path = os.path.join(self.setting.dataset_path, people)
            for device in setting.device_list:
                device_path = os.path.join(people_path, device)
                for scene in os.listdir(device_path):
                    scene_re = re.findall(r'[a-z]*_([a-z]*).*', scene)[0]
                    if scene_re not in setting.scene_list:
                        continue
                    scene_path = os.path.join(device_path, scene)
                    for wav_filename in os.listdir(scene_path):
                        if not (wav_filename[-3:] == 'wav' and wav_filename[-5] in label_map):
                            continue
                        label = label_map[wav_filename[-5]]
                        son_scene = re.findall(r'_(\D.?)_\d*_\d\.wav$', wav_filename)
                        son_scene = son_scene[0] if son_scene else 'None'
                        wav_path = os.path.join(scene_path, wav_filename)
                        voice_array, sample_rate = librosa.load(wav_path, sr=48000, mono=False)

                        # print('people:', people, 'device:', device, 'scene:', scene_re, 'son_scene:', son_scene)
                        people_n = setting.people_dic[people]
                        # for voice_segment in window_cut(voice_array, sample_rate, lens_win=200, step=100):
                        for voice_segment in middle_cut(voice_array, sample_rate, lens=400, pad_type='rand'):
                            try:
                                specgram = specgrams['mel'](voice_segment, sample_rate, 224, 112)
                            except:
                                break
                            self.sample_list.append({'voice_data': voice_segment, 'specgram': specgram,
                                                     'label': label, 'people': setting.people_dic[people],
                                                     'device': setting.device_dic[device],
                                                     'scene': setting.scene_dic[scene_re],
                                                     'son_scene': setting.son_scene_dic_dict[scene_re][son_scene]})
        random.shuffle(self.sample_list)

    def __len__(self):            # 返回整个数据集的大小
        return len(self.sample_list)

    def __getitem__(self, index):          # 根据索引index返回dataset[index]
        return self.sample_list[index]     # 返回该样本


