import torch
from torch.utils.data import DataLoader
from saveinf import Save
from model_select import model
from train import train_ch5
from plottool import plot_specgram, plot_wave_spec, plot_confusion_matrix
import os
import numpy as np
from k_fold import k_crossscene, k_nocrosspeople, k_crosspeople
from special_kfold import k_crossdevice_2, k_mixingdevice
import torch.nn as nn
import xlwt


def getinf_of_sample(sample_data, n, setting):  # 获得一个批次中的第n个样本的相关信息
    people = sample_data['people'][n]
    device = sample_data['device'][n]
    scene = sample_data['scene'][n]
    son_scene = sample_data['son_scene'][n]
    label = sample_data['label'][n]
    print('people: %s, device: %s, scene: %s, son_scene: %s, label: %s,' %
          (setting.people_list[people], setting.device_list[device], setting.scene_list[device],
           setting.son_scene_dict[setting.scene_list[scene]][son_scene], setting.label_list[label]))


def run():
    save_inf = Save()
    kfold_testacc = []
    count = 0
    for train_set, test_set, trainset_setting, testset_setting in k_nocrosspeople():  # 选择跨人还是不跨人
        train_iter = DataLoader(dataset=train_set, batch_size=340, shuffle=True)  # 载入训练集, 选择批量大小
        test_iter = DataLoader(dataset=test_set, batch_size=204, shuffle=True)  # 载入测试集
        print('训练集大小：', len(train_set), '测试集大小：', len(test_set))
        model_list = ['LeNet', 'vgg11', 'AlexNet', 'googlenet', 'mobilenet_v2',
                      'resnet18', 'RNN', 'GRU', 'LSTM', 'CRNN', 'LCZnet', 'Linear']  # 支持的模型列表
        net = model['resnet18'](num_classes=3)
        # net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        lr, num_epochs = 0.001, 200  # 定义学习率和epoch次数
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 定义优化算法
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义运行设备，显卡可用就用显卡

        test_result, test_acc = train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, testset_setting,
                                          save_inf, show=False)  # 训练及测试
        kfold_testacc.append(test_acc)
        count += 1

    print(kfold_testacc)
    print('平均测试集准确率：', np.average(kfold_testacc))
    print('最大：', np.max(kfold_testacc),
          '最小：', np.min(kfold_testacc),
          '标准差：', np.std(kfold_testacc))
    plot_confusion_matrix(save_inf.cat_fold_tensor('truth_list'), save_inf.cat_fold_tensor('pred_list'),
                          n_classes=3, savename=os.path.join(testset_setting.img_save_path, 'all.png'))  # 类别变时这里要改
    return np.average(kfold_testacc)


def acc_analysis(acc_list):
    _result = {
        '平均': np.average(acc_list),
        '最大': np.max(acc_list),
        '最小': np.min(acc_list),
        '标准差': np.std(acc_list)}
    print(_result)
    return _result


run()
#
# work_book = xlwt.Workbook(encoding='utf-8')
# sheet = work_book.add_sheet('sheet')
# results = []
# for i in range(10):
#     _result = run()
#     results.append(_result)
#
# print(results)
# results = np.array(results)
# ana_result = acc_analysis(results)
# for i, key in enumerate(ana_result):
#     sheet.write(0, i, ana_result[key])
#
# work_book.save(r'随机抽取漏气样本测试10次.xls')
