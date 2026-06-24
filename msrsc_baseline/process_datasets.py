import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

DATA_PATH = "../data"

# 预处理数据 1 加载原始数据 2 映射实体和关系到ID 3 保存映射文件 4 保存映射后的数据 5 创建过滤列表 6 计算实体概率分布
def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            # 头部实体，关系，尾部实体
            lhs, rel, rhs = line.strip().split('\t')
            # 都是set的所有没有重复的
            # 头实体
            entities.add(lhs)
            # 关系
            entities.add(rhs)
            # 尾部实体
            relations.add(rel)
        to_read.close()
    # entity映射
    # {
    #     'entity1': 0,
    #     'entity2': 1, 
    # }
    # 列表生成器的写法
    # entities_to_id是一个字典，键是实体名称，值是对应的唯一整数ID。
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    # relation映射
    # relations_to_id是一个字典，键是关系名称，值是对应的唯一整数ID。
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    # 关系的数量
    n_relations = len(relations)
    # 实体的数量
    n_entities = len(entities)
    # 创建data/wn18rr等文件夹
    os.makedirs(os.path.join(DATA_PATH, name))

    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    # ['train', 'valid', 'test']
    # 这段代码的作用是将原始文本三元组转换为数值 ID 格式，并序列化保存为 pickle 文件。
    # 构建基于id的三元组表示的数据集，结果保存到对应的pickle文件中
    # 因为有三个文件，所有要生成三个pickle文件,test,train,valid
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        # lhs 为left hand side 实体 rhs 为 right hand side 实体
        # examples是一个列表，里面的每个元素是一个三元组，三元组中的每个元素都是对应的ID,结构为[[lhs_id, rel_id, rhs_id], ...]
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    # 这段代码的作用是构建过滤列表，用于链接预测的 Filtered 评估。
    # 就是建立一个字典，记录每个（头实体，关系）对应的所有尾实体，以及每个（尾实体，关系）对应的所有头实体
    # 评估的时候将这些对应的得分写为无穷小，就可以达到过滤的效果
    # to_skip_final的结构如下:{''lhs': {(rhs_id, rel_id): [lhs_id1, lhs_id2, ...], ...}, 'rhs': {(lhs_id, rel_id): [rhs_id1, rhs_id2, ...], ...}}
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            # 所有的包括训练接测试集和验证集
            # 逆关系，[（头实体，关系）].add(尾实体)
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            # 正关系
            to_skip['rhs'][(lhs, rel)].add(rhs)
    # defaultdict 在 pickle 序列化/反序列化时可能有兼容性问题，普通 dict 更安全
    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    # 计算实体的概率分布
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        # 作为头实体出现的次数
        counters['lhs'][lhs] += 1
        # 作为尾实体出现的次数
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)

    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    datasets = ['WN18RR', 'FB237', 'YAGO3-10','UMLS','KINSHIP']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    '../src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise