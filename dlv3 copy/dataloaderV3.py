import os
from torch.utils.data import Dataset
import re
import random
from preprocess import window_cut, middle_cut, specgrams
import librosa

# 为了更模块化，数据读取分为两步
# 第一步，根据需求只读取文件的路径，保存
# 第二部，根据上一步的路径直接读取文件
class MyDatasetV2(Dataset):            # 继承Dataset
    def __init__(self, setting):     # __init__是初始化该类的一些基础参数
        self.setting = setting       # 相关设置均转移至setting.py文件里
        self.fileinf_list = []
        self.sample_list = []
        label_map = {'0': 'continue', '1': 0, '2': 'continue', '3': 1, '4': 2}
        for people in setting.people_list:          # 读取文件信息，但不读取文件
            print('正在读取%s的数据' % people)
            for device in setting.device_list:
                device_path = os.path.join(self.setting.dataset_path, people, device)
                for scene in os.listdir(device_path):
                    scene_re = re.findall(r'[a-z]*_([a-z]*).*', scene)[0]
                    if scene_re not in setting.scene_list:
                        continue
                    scene_path = os.path.join(device_path, scene)
                    for wav_filename in os.listdir(scene_path):
                        if not (wav_filename[-3:] == 'wav'):
                            continue
                        label = label_map[wav_filename[-5]]
                        if label == 'continue':
                            continue
                        son_scene = re.findall(r'_(\D.?)_\d*_\d\.wav$', wav_filename)
                        son_scene = son_scene[0] if son_scene else 'None'
                        self.fileinf_list.append({'filepath': os.path.join(scene_path, wav_filename),
                                                  'label': label,
                                                  'people': setting.people_dic[people],    # 对文件信息进行编码
                                                  'device': setting.device_dic[device],
                                                  'scene': setting.scene_dic[scene_re],
                                                  'son_scene': setting.son_scene_dic_dict[scene_re][son_scene]})
        random.shuffle(self.fileinf_list)                       # 打乱
        self.fileinf_list = file_filter(self.fileinf_list)      # 过滤
        for sample in self.fileinf_list:                        # 读取文件并且预处理
            voice_array, sample_rate = librosa.load(sample['filepath'], sr=48000)
            # for voice_segment in window_cut(voice_array, sample_rate, lens_win=200, step=100):
            for voice_segment in middle_samples(voice_array, sample_rate, lens=300, pad_type='rand'):
                # specgram = specgrams['mel'](voice_segment, sample_rate, 224, 112)
                specgram = specgrams['sff'](voice_segment, sample_rate, 224, 129)
                _sample_dict = sample.copy()
                _sample_dict['specgram'] = specgram
                _sample_dict['filepath'] = setting.path2num[_sample_dict['filepath']]
                self.sample_list.append(_sample_dict)
        random.shuffle(self.sample_list)

    def __len__(self):            # 返回整个数据集的大小
        return len(self.sample_list)

    def __getitem__(self, index):          # 根据索引index返回dataset[index]
        return self.sample_list[index]     # 返回该样本

# 数据过滤
def file_filter(fileinf_list):
    all = []
    louqi = {}
    # 1 遍历一遍，将样本集分几类：三种漏气程度下每个场景各一类，非漏气的全部作为一类
    for fileinf in fileinf_list:
        if fileinf['label'] != 0:
            all.append(fileinf)
        else:
            _key =  str(fileinf['scene'])
            if _key not in louqi.keys():
                louqi[_key] = [fileinf]
            else:
                louqi[_key].append(fileinf)
    all = all[:int(len(all)*(3/3))]
    # 2 从漏气各类中随机抽取 与非漏气类场景样本数量相同 的样本，并于非漏气样本集合并
    sum_louqi = 0
    for key in louqi:
        _len = int(len(louqi[key])*(3/3))
        sum_louqi += _len
        louqi[key] = random.sample(louqi[key], _len)
        all = all + louqi[key]
    print('漏气的文件数量有', sum_louqi)
    return all