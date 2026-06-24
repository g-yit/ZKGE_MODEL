import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import torch
from models import KBCModel


class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        # 通过三个pickle文件加载数据集
        # 就是转化为id后的三元组
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)
        # self.data['train'] 的内容举例:[[0,0,2], [0,1,3], [1,0,2], [2,2,4]]

        print(self.data['train'].shape)
        # [[0,0,2], [0,1,3], [1,0,2], [2,2,4]]
        # [2, 2, 4]
        # 计算每一列的最大值
        maxis = np.max(self.data['train'], axis=0)
        # [[0,0,2], [0,1,3], [1,0,2], [2,2,4]] maxis = [2, 2, 4]
        # n_entities 实体的适量
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        # 关系的数量
        self.n_predicates = int(maxis[1] + 1)
        # 真实的关系数量 (不包括反向关系)
        self.real_r = self.n_predicates
        self.relaions = []
        self.MRR = []
        self.hit1 = []

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        # to_skip的结构如下:{''lhs': {(rhs_id, rel_id): [lhs_id1, lhs_id2, ...], ...}, 'rhs': {(lhs_id, rel_id): [rhs_id1, rhs_id2, ...], ...}}

    # 计算测试集中每个实体的出现权重，用于加权评估。
    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['test'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1
        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        # 返回的结果的结构为：[w_0, w_1, w_2, ..., w_n] n为实体数量
        return w

    # 获取每个实体的原始出现频率
    def get_frequencies(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1
        return appear_list

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        # 关系变为了2倍
        # 建立反向关系
        sdata = self.data['train']
        r_c = self.real_r
        copy = np.copy(sdata)
        # 就是将头尾实体交换，并且关系加上 r_c（翻转关系）
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += r_c
        self.n_predicates = r_c*2
        # np.vstack 是 NumPy 的垂直堆叠函数，将多个数组按行（垂直方向拼接在一起。
        sdata = np.vstack((sdata, copy))
        # 三元组的数量变为了2倍
        return sdata

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        # 评估模型在指定数据集上的表现
        model.eval()
        # 得到数据集
        test = self.get_examples(split)
        # 将数据放在cuda中
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        # 评估的的模式，是预测头实体还是尾实体，还是两者都评估
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            # 这里的examples是数据集中的三元组结构为[[lhs_id, rel_id, rhs_id], ...],为所有的测试集的数据
            # n_queries 表示要评估的查询数量，-1表示评估所有查询，默认为-1
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
                
            if m == 'lhs':
                # 预测头实体
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.real_r
            # ranks 是一个长度为 n_queries 的一维数组，表示每个查询的正确实体在排序中的位置
            # 意思就是测试了这么多，每一个测试的三元组，正确的实体在所有实体中的排名是多少[1,2,3,9,20,...]
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    # examples为数据集中的三元组结构为[[lhs_id, rel_id, rhs_id], ...]
    def get_pos(self, examples):
        # 这个pos表示的是positive samples的意思
        dic_tr ={}
        dic_hr = {}
        dic_t = {}
        dic_h = {}
        for i in examples:
            # tail-relation: 给定头实体，找所有 (尾, 关系)
            # 这些(尾, 关系)共享同一个头实体
            # .item() 是 PyTorch 的方法，用于将只包含一个元素的张量转换为 Python 标量
            dic_tr[i[0].item()] = []
            # head-relation: 给定尾实体，找所有 (头, 关系)
            # 这些(头, 关系)共享同一个尾实体
            dic_hr[i[2].item()] = []
            # tail: 给定 (头, 关系)，找所有尾实体，这些尾实体共享同一个(头, 关系)
            dic_t[(i[0].item(), i[1].item())] =[]
            # head: 给定 (尾, 关系)，找所有头，这些头共享同一个(尾, 关系)
            dic_h[(i[2].item(), i[1].item())] = []
        for i in examples:
            dic_tr[i[0].item()].append([i[2].item(), i[1].item()])
            dic_hr[i[2].item()].append([i[0].item(), i[1].item()])
            dic_t[(i[0].item(), i[1].item())].append(i[2].item())
            dic_h[(i[2].item(), i[1].item())].append(i[0].item())
        return dic_tr, dic_hr, dic_h, dic_t

    def get_shape(self):
        # 得到实体数量，关系数量，实体数量
        return self.n_entities, self.n_predicates, self.n_entities