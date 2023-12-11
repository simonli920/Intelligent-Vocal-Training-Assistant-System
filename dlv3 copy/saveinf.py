import torch
class Save():
    def __init__(self):
        # 保存每一折的预测值，预测值对应的真实值，损失数列，训练集准确率数列，测试集准确率数列
        self.inf_dir = {'pred_list': [], 'truth_list': [], 'loss_list': [], 'trainacc_list': [], 'testacc_list': []}

    def append(self, fold_inf, type='None'):
        # type\: 'pred_list', 'truth_list', 'loss_list', 'trainacc_list', 'testacc_list'
        if type not in self.inf_dir.keys():
            print('wrong type!!!')
        else:
            self.inf_dir[type].append(fold_inf)

    def get_fold_inf(self, k_of_fold = 1, type='None'):
        return self.inf_dir[type][k_of_fold-1]

    def cat_fold_tensor(self, type='None', to_numpy=False):
        # type\: 'pred_list', 'truth_list'
        if type not in ['pred_list', 'truth_list']:
            print('wrong type!!!')
        else:
            _allpred = torch.tensor([])
            for pred in self.inf_dir[type]:
                _allpred = torch.cat((_allpred, pred))
            if to_numpy==True:
                return _allpred.numpy()
            else:
                return _allpred



