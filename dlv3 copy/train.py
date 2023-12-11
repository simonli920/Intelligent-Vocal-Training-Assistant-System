import torch
import time
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from plottool import plot_confusion_matrix,plot_wave_spec

def getinf_of_sample(sample_data, n, setting):  # 获得一个批次中的第n个样本的相关信息
    people = sample_data['people'][n]
    device = sample_data['device'][n]
    scene = sample_data['scene'][n]
    son_scene = sample_data['son_scene'][n]
    label = sample_data['label'][n]
    print('people: %s, device: %s, scene: %s, son_scene: %s, label: %s,' %
          (setting.people_list[people], setting.device_list[device], setting.scene_list[scene],
           setting.son_scene_dict[setting.scene_list[scene]][son_scene], setting.label_list[label]))


def train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, testset_setting, saveinf, show=True):
    # 训练函数，参数依次为网络模型，训练集，测试集，优化器，计算设备，epoch次数
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    loss_array = []
    test_loss_array = []
    accuracy_array_train = []
    accuracy_array_test = []
    _, _, _, train_err = evaluate_accuracy(train_iter, net)
    _, _, _, test_err = evaluate_accuracy(test_iter, net)
    loss_array.append(train_err)
    test_loss_array.append(test_err)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for data in train_iter:
            X = data['specgram']
            y = data['label']
            X = X.type(torch.FloatTensor)
            y = y.long()
            X = X.to(device)
            y = y.to(device)


            y_hat = net(X)
            # print('X.shape:', X.shape, 'y.shape:', y.shape, 'y_hat.shape:', y_hat.shape)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        loss_rate = train_l_sum / batch_count
        train_acc = train_acc_sum / n
        truth_tensor, pred_tensor, test_acc, test_err = evaluate_accuracy(test_iter, net)


        loss_array.append(loss_rate)
        test_loss_array.append(test_err)
        accuracy_array_train.append(train_acc)
        accuracy_array_test.append(test_acc)


        if epoch % 1 == 0:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, loss_rate, train_acc, test_acc, time.time() - start))
        if train_acc > 0.95:
            if len(accuracy_array_test) > 15:
                last_10 = np.array(accuracy_array_test[-10:])
                std = np.std(last_10)
                print(std)
                if std < 0.001:
                    truth_tensor, pred_tensor, test_acc, test_err = evaluate_accuracy(test_iter, net, testset_setting,
                                                                            plot=True, show_wrong_spec=False)
                    saveinf.append(pred_tensor, 'pred_list')
                    saveinf.append(truth_tensor, 'truth_list')
                    break
    if show:
        x_oder = np.arange(0, len(loss_array))
        plt.plot(x_oder, loss_array, label='train')
        plt.plot(x_oder, test_loss_array, label='eval')
        plt.legend(loc='best')
        plt.title('Running loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        # plt.figure()
        # plt.plot(x_oder, accuracy_array_train, color='red', linewidth=2.0, linestyle='-')
        #
        # plt.figure()
        # plt.plot(x_oder, accuracy_array_test, color='red', linewidth=2.0, linestyle='-')
        plt.show()
    return pred_tensor, test_acc


def evaluate_accuracy(data_iter, net, setting=None, device=None, plot=False, show_wrong_spec=False):
    test_l_sum, test_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
    # 计算在测试集上的准确率
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    pred_tensor = torch.tensor([]).to(device)
    truth_tensor = torch.tensor([])
    loss = torch.nn.CrossEntropyLoss()
    loss_array = []
    with torch.no_grad():
        for data in data_iter:
            X = data['specgram']
            y = data['label']
            truth_tensor = torch.cat((truth_tensor, y))
            X = X.type(torch.FloatTensor)
            y = y.long()
            X = X.to(device)
            y = y.to(device)
            net.eval()                                     # 评估模式, 这会关闭dropout
            # print('test X', X.shape,)
            y_hat = net(X)
            l = loss(y_hat, y)
            test_l_sum += l.cpu().item()
            y_hat = y_hat.argmax(dim=1)
            pred_tensor = torch.cat((pred_tensor, y_hat))
            # print(y, '->', y_hat)

            acc_sum += (y_hat == y).float().sum().cpu().item()
            net.train()                                   # 改回训练模式
            n += y.shape[0]
            batch_count += 1

            if show_wrong_spec:
                waveform = data['voice_data']
                y_hat = y_hat.cpu()
                for i in range(y.shape[0]):
                    if y[i] != y_hat[i]:
                        plot_wave_spec(waveform[i], X[i, 0])
                        getinf_of_sample(data, i, setting)
                        print(y[i].item(), '->', y_hat[i].item())
    loss_rate = test_l_sum / batch_count
    if plot == True:
        count = 1
        filenames = os.listdir(setting.img_save_path)
        while True:
            if str(count)+r'.png' not in filenames:
                break
            count += 1
        plot_confusion_matrix(truth_tensor, pred_tensor.cpu(), n_classes=3,
                              savename=os.path.join(setting.img_save_path, str(count)+r'.png'))
        # 类别变时这里要改
    # 依次返回真实值，预测值，准确率，误差
    return truth_tensor, pred_tensor.cpu(), acc_sum / n, loss_rate

