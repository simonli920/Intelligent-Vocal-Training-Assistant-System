import time
from dataloaderV2 import MyDataset
from dataloaderV1 import MyDataset_v0
from setting import Setting, read_filename, path2num, dataset_path
from torch.utils.data import ConcatDataset, Subset
import re
import random


# 跨设备
# 读入专业麦的所有文件列表，并随机分配设备类型：其中1/k为测试设备，其余为训练设备
# 按上面分配组合路径读入文件，作为训练集和测试集
def k_crossdevice_2():
    device_list = ['mic', 'k30s', 's20', 'neck_MIC', 'neck_VI']
    k = len(device_list)
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(跨设备不跨人无重复样本)", time.localtime())
    gen_setting = Setting(img_save_dir=img_dir)
    mic_file = [path for path in read_filename(dataset_path) if 'mic' in path]
    testset_num = int(len(mic_file)/k)
    paths_k = []
    for i in range(k):
        paths_k.append([mic_file.pop(random.randint(0, len(mic_file) - 1)) for i in range(testset_num)])

    p_datasets = []
    start_time = time.time()
    for i in range(k):
        for j in range(testset_num):
            paths_k[i][j] = re.sub('mic', device_list[i], paths_k[i][j])
        for p in range(10):
            print(random.choice(paths_k[i]))
        p_dataset = MyDataset_v0(paths_k[i], path2num)
        p_datasets.append(p_dataset)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))

    for i in range(k):
        print(len(p_datasets[i]))

    for i in range(k):
        test_set = p_datasets[i]
        train_set = ConcatDataset(p_datasets[:i] + p_datasets[i + 1:])
        yield train_set, test_set, gen_setting, gen_setting



# 跨设备
def k_crossdevice_1(k=10):
    device_list = ['mic', 'k30s', 's20', 'neck_MIC', 'neck_VI']
    test_num = 1
    k = int(len(device_list))
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(跨设备同时跨人)", time.localtime())
    p_datasets = []
    start_time = time.time()
    for device in device_list:
        p_setting = Setting(device_list=[device], img_save_dir=img_dir)
        p_dataset = MyDataset(setting=p_setting)
        p_datasets.append(p_dataset)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))
    for i in range(k):
        test_set = ConcatDataset(p_datasets[i * test_num:(i + 1) * test_num])
        train_set = ConcatDataset(p_datasets[:i * test_num] + p_datasets[(i + 1) * test_num:])
        testset_setting = Setting(device_list=device_list[i*test_num:(i+1)*test_num],
                                  img_save_dir=img_dir)
        trainset_setting = Setting(device_list=device_list[:i*test_num]+device_list[(i+1)*test_num:],
                                   img_save_dir=img_dir)
        yield train_set, test_set, trainset_setting, testset_setting


# 混合设备
def k_mixingdevice(k=10):
    device_list = ['mic', 'k30s', 's20']
    num_device = len(device_list)
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(混合设备不跨人)", time.localtime())
    gen_setting = Setting(img_save_dir=img_dir)
    mic_file = [path for path in read_filename(dataset_path) if 'mic' in path]
    filenums_of_device = int(len(mic_file)/num_device)

    paths_k = []
    paths_all = []
    for i in range(num_device):
        paths_k.append([mic_file.pop(random.randint(0, len(mic_file) - 1)) for i in range(filenums_of_device)])

    start_time = time.time()
    for i in range(num_device):
        for j in range(filenums_of_device):
            _path = re.sub('mic', device_list[i], paths_k[i][j])   # 替换掉路径中的设备
            paths_all.append(_path)
    print('专业麦数据集大小', len(paths_all))
    full_set = MyDataset_v0(paths_all, path2num)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))

    dataset_length = len(full_set)  # 选择训练集占比
    print('数据集大小', dataset_length)
    testset_size = int(dataset_length / k)
    for i in range(k):
        test_set = Subset(full_set, range(testset_size * i, testset_size * (i + 1)))  # 划分测试集
        left_index = list(range(testset_size * i))
        right_index = list(range(testset_size * (i + 1), dataset_length))
        train_set = Subset(full_set, left_index + right_index)  # 划分训练集
        yield train_set, test_set, gen_setting, gen_setting

