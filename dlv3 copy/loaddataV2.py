import os
import re
import random
from preprocess import window_cut, middle_cut, specgrams
import librosa

# 为了更模块化，数据读取分为两步
# 第一步，根据需求只读取文件的路径，保存
# 第二部，根据上一步的路径直接读取文件

def mydatasetV2(setting):     # __init__是初始化该类的一些基础参数
    setting = setting       # 相关设置均转移至setting.py文件里
    fileinf_list = []
    sample_list = []
    label_map = {'0': 'continue', '1': 0, '2': 'continue', '3': 1, '4': 2}
    for people in setting.people_list:  # 读取文件信息，但不读取文件
        print('正在读取%s的数据' % people)
        for device in setting.device_list:
            device_path = os.path.join(setting.dataset_path, people, device)
            for scene in os.listdir(device_path):
                scene_re = re.findall(r'^[^_]*_([^_]*).*', scene)[0]
                if scene_re not in setting.scene_list:
                    continue
                scene_path = os.path.join(device_path, scene)
                for wav_filename in os.listdir(scene_path):
                    if not (wav_filename[-3:] == 'wav'):
                        continue
                    old_label = wav_filename[-5]
                    label = label_map[old_label]  # 标签映射
                    if label == 'continue':
                        continue
                    result = re.findall(r'^[^_]*_[^_]*_([^_]*).*', wav_filename)
                    son_scene = result[0] if result else 'None'
                    fileinf_list.append({'filepath': os.path.join(scene_path, wav_filename),
                                              'label': label,
                                              'old_label': int(old_label),
                                              'people': people,  # 读入文件信息
                                              'device': device,
                                              'scene': scene_re,
                                              'son_scene': son_scene})
    random.shuffle(fileinf_list)  # 打乱
    fileinf_list = file_filter2(fileinf_list, label_map)  # 过滤
    for sample in fileinf_list:                        # 读取文件并且预处理
        voice_array, sample_rate = librosa.load(sample['filepath'], sr=8000, mono=False)
        voice_array = voice_array[0]
        if sample['label'] == 2 and sample['scene'] == 'payin':
            voice_array_list = window_cut(voice_array, sample_rate, lens_win=setting.window_len, step=setting.window_len)
        else:
            voice_array_list = middle_cut(voice_array, sample_rate, lens=setting.window_len, pad_type='none')
        for voice_segment in voice_array_list:
            # spec_list = ['stft', 'mel', 'mfccs', 'sff', 'lpc']
            specgram = specgrams['mfccs'](voice_segment, sample_rate,
                                        setting.n_specgram, setting.hop_specgram)
            _sample_dict = sample.copy()
            _sample_dict['mfccs'] = specgram.flatten()
            sample_list.append(_sample_dict)
    random.shuffle(sample_list)
    return sample_list


# 数据过滤
def file_filter(fileinf_list):
    _all = []
    louqi = {}
    # 1 遍历一遍，将样本集分几类：三种漏气程度下每个场景各一类，非漏气的全部作为一类
    for fileinf in fileinf_list:
        if fileinf['label'] != 0:
            _all.append(fileinf)
        else:
            _key =  str(fileinf['scene'])
            if _key not in louqi.keys():
                louqi[_key] = [fileinf]
            else:
                louqi[_key].append(fileinf)
    # 2 从漏气各类中随机抽取 与非漏气类场景样本数量相同 的样本，并于非漏气样本集合并
    sum_louqi = 0
    for key in louqi:
        _len = int(len(louqi[key])/3)
        sum_louqi += _len
        louqi[key] = random.sample(louqi[key], _len)
        _all = _all + louqi[key]
    print('不漏气的文件数量有', sum_louqi)
    return _all


# 数据过滤, 若选择多类作为一类标签，则按照一定比例从其各场景中随机抽取样本，使每个类别的数量大致相同
def file_filter2(fileinf_list, label_map):
    _label_map = {}
    for key in label_map:
        if label_map[key] != 'continue':
            _label_map[key] = label_map[key]
    all_dict = {}
    _all = []                # 用于存放过滤完的样本数组
    # 1 遍历一遍，生成将数据集整理成一个两层字典，第一层按老标签分，第二层按场景分，最下面是样本列表
    for fileinf in fileinf_list:
        _label = str(fileinf['old_label'])
        _scene = str(fileinf['scene'])
        if _label not in all_dict.keys():
            all_dict[_label] = {}
        if _scene not in all_dict[_label].keys():
            all_dict[_label][_scene] = [fileinf]
        else:
            all_dict[_label][_scene].append(fileinf)

    counts = [0] * len(_label_map)
    for olb in _label_map:
        counts[_label_map[olb]] += 1
    ratio_list = [1 / n for n in counts if n]
    # 2 按照一定比例从其各场景中随机抽取样本
    for olb in all_dict:
        _sum = 0
        for scene in all_dict[olb]:
            _len = int(len(all_dict[olb][scene]) * ratio_list[_label_map[olb]])
            _sum += _len
            _samples = random.sample(all_dict[olb][scene], _len)
            _all = _all + _samples
        print('标签为', olb, '的样本数量有', _sum)
    return _all