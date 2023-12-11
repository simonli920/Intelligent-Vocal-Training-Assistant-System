import time
from dataloaderV2 import MyDataset
from dataloaderV3 import MyDatasetV2
from setting import Setting
from torch.utils.data import ConcatDataset, Subset

# 单设备跨人
def k_crosspeople(k=20):
    people_list = ['gm', 'gys', 'hjk', 'lab', 'lhl',
                   'lm', 'lq', 'lst', 'lyh', 'lzb',
                   'qy', 'sjy', 'wgn', 'yhs', 'ykn',
                   'ysd', 'yxs', 'yyp', 'zwj', 'zym']
    test_num = int(len(people_list)/k)
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(跨人)", time.localtime())
    p_settings = []
    p_datasets = []
    start_time = time.time()
    for people in people_list:
        p_setting = Setting(device_list=['mic'], people_list=[people], exclude_people=False,
                            img_save_dir=img_dir)
        p_dataset = MyDataset(setting=p_setting)
        p_settings.append(p_setting)
        p_datasets.append(p_dataset)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))
    for i in range(k):
        test_set = ConcatDataset(p_datasets[i*test_num:(i+1)*test_num])
        train_set = ConcatDataset(p_datasets[:i*test_num]+p_datasets[(i+1)*test_num:])
        testset_setting = Setting(device_list=['mic'], people_list=people_list[i*test_num:(i+1)*test_num],
                                   exclude_people=False, img_save_dir=img_dir)
        trainset_setting = Setting(device_list=['mic'], people_list=people_list[i*test_num:(i+1)*test_num],
                                   exclude_people=True, img_save_dir=img_dir)
        yield train_set, test_set, trainset_setting, testset_setting

# 多设备跨人，单设备测试
def k_crosspeople2(k=10):
    people_list = ['gm', 'gys', 'hjk', 'lab', 'lhl',
                   'lm', 'lq', 'lst', 'lyh', 'lzb',
                   'qy', 'sjy', 'wgn', 'yhs', 'ykn',
                   'ysd', 'yxs', 'yyp', 'zwj', 'zym']
    test_num = int(len(people_list)/k)
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(多设备跨人)", time.localtime())
    p_datasets = []
    p_datasets_test = []
    start_time = time.time()
    _d = ['mic', 'k30s', 's20', 'neck_VI']
    _dt = ['neck_MIC']
    for people in people_list:
        p_setting = Setting(device_list=_d,
                            people_list=[people], exclude_people=False,
                            img_save_dir=img_dir)
        p_dataset = MyDataset(setting=p_setting)
        p_datasets.append(p_dataset)
        p_setting_t = Setting(device_list=_dt,
                            people_list=[people], exclude_people=False,
                            img_save_dir=img_dir)
        p_dataset_t = MyDataset(setting=p_setting_t)
        p_datasets_test.append(p_dataset_t)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))
    for i in range(k):
        test_set = ConcatDataset(p_datasets_test[i*test_num:(i+1)*test_num])
        train_set = ConcatDataset(p_datasets[:i*test_num]+p_datasets[(i+1)*test_num:]
                                  + p_datasets_test[:i*test_num]+p_datasets_test[(i+1)*test_num:])
        testset_setting = Setting(device_list=_d, people_list=people_list[i*test_num:(i+1)*test_num],
                                   exclude_people=False, img_save_dir=img_dir)
        trainset_setting = Setting(device_list=_d, people_list=people_list[i*test_num:(i+1)*test_num],
                                   exclude_people=True, img_save_dir=img_dir)
        yield train_set, test_set, trainset_setting, testset_setting


# 不跨人
def k_nocrosspeople(k=10):
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(不跨人)", time.localtime())
    full_set_setting = Setting(device_list=['mic'],
                               people_list=[],
                               exclude_people=True,
                               img_save_dir=img_dir)
    start_time = time.time()
    full_set = MyDatasetV2(setting=full_set_setting)                                    # 根据setting的信息读入数据集
    print('数据及大小',len(full_set))
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))
    dataset_length = len(full_set)                                                    # 选择训练集占比
    testset_size = int(dataset_length/k)
    for i in range(k):
        test_set = Subset(full_set, range(testset_size*i, testset_size*(i+1)))        # 划分测试集
        left_index = [j for j in range(testset_size*i)]
        right_index = [k for k in range(testset_size*(i+1),dataset_length)]
        train_set = Subset(full_set, left_index+right_index)                          # 划分训练集
        yield train_set, test_set, full_set_setting, full_set_setting

# 不跨人,使用数据增强
# 只在训练集使用数据增强，测试集使用原始样本
# 只在开始一次性读入原始样本和生成增强样本，后续交叉验证不生成新的增强样本
# 处理流程如下：
# 1 生成原始样本和增强样本，将原始样本和来自次原始样本的增强样本合在一个数组里，所有这样的数组合成一个更大的数组
# 2 按交叉验证切分方式划分训练集和测试集
# 3 选取训练集部分的样本数组集，摊平为一个样本数组
# 4 只选取测试集部分中的原始样本，组成一个样本数组
# 5 3和4中的样本数组各送到dataload类里面生成dataset对象，以供后面使用
def k_nocrosspeople2(k=10):
    pass



# 跨场景
def k_crossscene():
    scene_list = ['vowellong', 'vowelshort', 'fade', 'staccato']
    son_scene_dict = {'vowellong': ['a', 'e', 'i', 'o', 'u'],
                      'vowelshort': ['a', 'e', 'i', 'o', 'u'],
                      'fade': ['a1', 'a0', 'e1', 'e0', 'i1', 'i0', 'o1', 'o0', 'u1', 'u0'],
                      'staccato': ['do', 'mi', 'so']}
    test_num = 1
    k = int(len(scene_list))
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(跨场景)", time.localtime())
    p_datasets = []
    start_time = time.time()
    for scene in scene_list:
        p_setting = Setting(device_list=['mic'],
                            scene_list=[scene], exclude_scene=False,
                            img_save_dir=img_dir)
        p_dataset = MyDataset(setting=p_setting)
        p_datasets.append(p_dataset)
    print('读取数据用时：%.3f 秒' % (time.time() - start_time))
    for i in range(k):
        test_set = ConcatDataset(p_datasets[i*test_num:(i+1)*test_num])
        train_set = ConcatDataset(p_datasets[:i*test_num]+p_datasets[(i+1)*test_num:])
        testset_setting = Setting(device_list=['mic'],
                                  scene_list=scene_list[i*test_num:(i+1)*test_num], exclude_scene=False,
                                  img_save_dir=img_dir)
        trainset_setting = Setting(device_list=['mic'],
                                   scene_list=scene_list[i*test_num:(i+1)*test_num],exclude_scene=True,
                                   img_save_dir=img_dir)
        yield train_set, test_set, trainset_setting, testset_setting


# 跨设备
def k_crossdevice_1(k=10):
    device_list = ['mic', 'k30s', 's20', 'neck_MIC', 'neck_VI']
    test_num = 1
    k = int(len(device_list))
    img_dir = time.strftime("%Y年%m月%d日%H时%M分(跨设备不跨人有重复样本)", time.localtime())
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


