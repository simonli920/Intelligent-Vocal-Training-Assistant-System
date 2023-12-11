from dataloaderV2 import MyDataset
from setting import Setting

# 统计数据集各类别，各场景的个数
def dataset_statistics(dataset):
    label_count = {}
    for sample in dataset:
        label = str(sample['label'])
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
    print(label_count)

full_set_setting = Setting(device_list=['mic'], scene_list=[], exclude_scene=True)  # 设置数据集读入范围
full_set = MyDataset(setting=full_set_setting)                     # 根据setting的信息读入数据集
dataset_statistics(full_set)
