import os


# 此类记录着数据集的读取范围信息, 旨在每当数据集源改动时，只需修改此类
class Setting():
    def __init__(self, device_list=['mic'], people_list=[], scene_list=[],
                 exclude_people: bool = True, exclude_scene: bool = True, exclude_device: bool = False,
                 feature_select=['mel_specgram'], model='resnet18',
                 img_save_dir=None):
        self.dataset_path = dataset_path
        self.device_list = ['mic', 'k30s', 's20', 'neck_MIC', 'neck_VI']
        self.people_list = ['gm', 'gys', 'hjk', 'lab', 'lhl',
                            'lm', 'lq', 'lst', 'lyh', 'lzb',
                            'qy', 'sjy', 'wgn', 'yhs', 'ykn',
                            'ysd', 'yxs', 'yyp', 'zwj', 'zym']
        self.scene_list = ['vowellong', 'vowelshort', 'fade', 'staccato']
        self.son_scene_dict = {'vowellong': ['a', 'e', 'i', 'o', 'u'],
                               'vowelshort': ['a', 'e', 'i', 'o', 'u'],
                               'fade': ['a1', 'a0', 'e1', 'e0', 'i1', 'i0', 'o1', 'o0', 'u1', 'u0'],
                               'staccato': ['do', 'mi', 'so']}
        self.label_list = ['闭合不严', '正常闭合', '过度闭合']
        # self.label_list = ['严重不严', '中等不严', '轻微不严', '正常闭合', '过度闭合']
        self.path2num = path2num
        self.people_dic = list2dic(self.people_list)
        self.device_dic = list2dic(self.device_list)
        self.scene_dic = list2dic(self.scene_list)
        self.label_dic = list2dic(self.label_list)
        self.son_scene_dic_dict = {}
        for scene in self.son_scene_dict.keys():
            self.son_scene_dic_dict[scene] = list2dic(self.son_scene_dict[scene])


        self.device_list = device_list
        if exclude_people:
            [self.people_list.remove(i) for i in people_list]
        else:
            self.people_list = people_list
        if exclude_scene:
            [self.scene_list.remove(i) for i in scene_list]
        else:
            self.scene_list = scene_list

        self.img_save_path = r'saveimg'
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        if img_save_dir:
            self.img_save_path = os.path.join(self.img_save_path, img_save_dir)
            if not os.path.exists(self.img_save_path):
                os.makedirs(self.img_save_path)

        self.window_len = 200     # 切分窗口长度，单位ms
        self.step = 100           # 切分窗口滑动步长，单位ms

    def inf_encode(self, ):
        pass


# 递归读取文件夹下所有文件列表
def read_filename(path):
    file_list = []
    files = os.listdir(path)               # 获取文件夹中文件和目录列表
    for f in files:
        f_path = os.path.join(path,f)
        if not os.path.isdir(f_path):      # 判断是否是文件夹
            file_list.append(f_path)
        else:
            for p in read_filename(f_path):
                file_list.append(p)        # 递归调用本函数
    return file_list


# 列表转换成字典，键为列表元素，值为列表序号
def list2dic(input_list):
    dic = {}
    for i,item in enumerate(input_list):
        dic[item] = i
    return dic

dataset_path = r'D:\code\python\voice coach\data\old_data\new_label_data_all'
path2num = list2dic(read_filename(dataset_path))

