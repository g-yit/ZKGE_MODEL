import ast
import os
import json
import argparse
import numpy as np
import pickle

import torch
from torch import optim
from torch import nn
from datasets import Dataset
from models import *
from regularizers import *
from optimizers import KBCOptimizer

datasets = ['WN18RR', 'FB237', 'YAGO3-10','UMLS','KINSHIP']

parser = argparse.ArgumentParser(
    description="Tensor Factorization for Knowledge Graph Completion"
)

# 数据集选择
parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)
# 模型选择
parser.add_argument(
    '--model', type=str, default='CP'
)
# 正则化选择
parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument('--name', type=str, default='WN18RR')
parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true', default=True)
parser.add_argument('--no_ce_weight', action='store_true', help='Disable entity-frequency class weights.')
parser.add_argument('--ce_weight_source', choices=['test', 'train'], default='test',
                    help='Source split for CE class weights. Default keeps baseline behavior.')
parser.add_argument('-path', '--save_path', type=str, default='./logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

parser.add_argument(
    '--negative_sample_size', default=200, type=int,
    help="negative sample size"
)
parser.add_argument('--out_size', default=4000, type=int, help="out size")
parser.add_argument("--min_lr", default=5e-5, type=float, help='min learning rate')

# conv模型参数
parser.add_argument("--input_drop", default=0.4, type=float, help="Dropout on input layer")
parser.add_argument("--hidden_drop", default=0.3, type=float, help="Dropout on hidden layer")
parser.add_argument("--feature_map_drop", default=0.3, type=float, help="Dropout on feature map")
parser.add_argument("--weight_decay", default=5e-8, type=float)
parser.add_argument("--factor", default=0.8, type=float)
parser.add_argument("--verbose", default=1, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument('--output_channel', dest="output_channel", default=20, type=int, help='Number of output channel')
parser.add_argument('--k_w', dest="k_w", default=10, type=int, help='Width of the reshaped matrix')
parser.add_argument('--k_h', dest="k_h", default=20, type=int, help='Height of the reshaped matrix')
parser.add_argument('--seed', type=int, dest="seed", default='2022', help='random seed')
parser.add_argument("--active_fn", default="relu", help="activation function for the model")
parser.add_argument("--init_fn", default="xavier_normal", help="initialization function for the model")
parser.add_argument("--filter_size_list", default=[(1, 5, 1, 2), (3, 3), (1, 9)],
                    help="卷积列表，格式为[(h,w,dh,dw),(h,w),(h,w)]")

# Context-guided scale routing
parser.add_argument("--use_scale_router", action="store_true",
                    help="Enable context-guided scale routing over convolution branches.")
parser.add_argument("--module_warmup_epochs", default=0, type=int,
                    help="Epochs during which new modules are identity-preserving.")
parser.add_argument("--module_ramp_epochs", default=1, type=int,
                    help="Epochs used to ramp new module strength after warmup.")
parser.add_argument("--router_hidden", default=0, type=int,
                    help="Hidden size of the scale router; 0 means embedding_dim.")
parser.add_argument("--router_dropout", default=0.1, type=float)
parser.add_argument("--router_temperature", default=1.0, type=float)
parser.add_argument("--router_min_branch_weight", default=0.0, type=float,
                    help="Lower bound for each branch weight, useful for stable early training.")
parser.add_argument("--router_residual_init", default=0.10, type=float,
                    help="Initial residual strength for branch routing.")
parser.add_argument("--use_router_residual", action="store_true",
                    help="Use baseline-preserving residual branch gains instead of direct softmax weights.")

# Relation-conditioned evidence memory
parser.add_argument("--use_rcem", action="store_true",
                    help="Enable relation-conditioned structural evidence memory.")
parser.add_argument("--rcem_no_path", action="store_true",
                    help="Disable two-hop relation composition evidence.")
parser.add_argument("--rcem_no_type", action="store_true",
                    help="Disable implicit entity-role type evidence.")
parser.add_argument("--rcem_max_rules", default=8, type=int,
                    help="Maximum mined path rules kept for each relation.")
parser.add_argument("--rcem_max_candidates", default=32, type=int,
                    help="Maximum evidence candidates stored for each query.")
parser.add_argument("--rcem_min_rule_support", default=3, type=int,
                    help="Minimum support for a mined relation composition rule.")
parser.add_argument("--rcem_max_rule_degree", default=64, type=int,
                    help="Maximum local degree used while mining and applying path rules.")
parser.add_argument("--rcem_warmup_epochs", default=0, type=int,
                    help="Epochs during which evidence residuals are disabled.")
parser.add_argument("--rcem_ramp_epochs", default=5, type=int,
                    help="Epochs used to ramp evidence residual strength after warmup.")
parser.add_argument("--rcem_gate_hidden", default=0, type=int,
                    help="Hidden size of evidence gate; 0 means embedding_dim.")
parser.add_argument("--rcem_gate_dropout", default=0.05, type=float)
parser.add_argument("--rcem_path_strength", default=0.10, type=float,
                    help="Maximum residual logit strength for path evidence.")
parser.add_argument("--rcem_type_strength", default=0.04, type=float,
                    help="Maximum residual logit strength for type evidence.")
parser.add_argument("--rcem_path_gate_init", default=0.05, type=float,
                    help="Initial relation gate probability for path evidence.")
parser.add_argument("--rcem_type_gate_init", default=0.05, type=float,
                    help="Initial relation gate probability for type evidence.")

args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.no_ce_weight:
    ce_weight = None
elif args.do_ce_weight:
    if args.ce_weight_source == 'train':
        ce_weight = torch.Tensor(dataset.get_weight_from_split('train')).cuda()
    else:
        ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model = None
regularizer = None

def parse_list_argument(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}") from e


if args.model == 'MSDCSE':
    filter_size_list = parse_list_argument(args.filter_size_list) if isinstance(args.filter_size_list, str) else args.filter_size_list
    relation_context = dataset.build_relation_context() if (args.use_scale_router or args.use_rcem) else None
    rcem_context = None
    if args.use_rcem:
        rcem_context = dataset.build_rcem_context(
            max_rules_per_relation=args.rcem_max_rules,
            max_candidates_per_query=args.rcem_max_candidates,
            min_rule_support=args.rcem_min_rule_support,
            max_rule_degree=args.rcem_max_rule_degree,
            use_path=not args.rcem_no_path,
            use_type=not args.rcem_no_type,
        )
    model = MSDCSE(
        num_ent=dataset.get_shape()[0],
        num_rel=dataset.get_shape()[1],
        embedding_dim=args.rank,
        input_drop=args.input_drop,
        hidden_drop=args.hidden_drop,
        feature_map_drop=args.feature_map_drop,
        k_w=args.k_w,
        k_h=args.k_h,
        output_channel=args.output_channel,
        filter_size_list=filter_size_list,
        active_fn=args.active_fn,
        init_fn=args.init_fn,
        ce_weight=ce_weight,
        use_scale_router=args.use_scale_router,
        relation_context=relation_context,
        module_warmup_epochs=args.module_warmup_epochs,
        module_ramp_epochs=args.module_ramp_epochs,
        router_hidden=args.router_hidden if args.router_hidden > 0 else None,
        router_dropout=args.router_dropout,
        router_temperature=args.router_temperature,
        router_min_branch_weight=args.router_min_branch_weight,
        router_residual=args.use_router_residual,
        router_residual_init=args.router_residual_init,
        use_rcem=args.use_rcem,
        rcem_context=rcem_context,
        rcem_use_path=not args.rcem_no_path,
        rcem_use_type=not args.rcem_no_type,
        rcem_warmup_epochs=args.rcem_warmup_epochs,
        rcem_ramp_epochs=args.rcem_ramp_epochs,
        rcem_gate_hidden=args.rcem_gate_hidden if args.rcem_gate_hidden > 0 else None,
        rcem_gate_dropout=args.rcem_gate_dropout,
        rcem_path_strength=args.rcem_path_strength,
        rcem_type_strength=args.rcem_type_strength,
        rcem_path_gate_init=args.rcem_path_gate_init,
        rcem_type_gate_init=args.rcem_type_gate_init,
    )
    model.init()

else:
    exec('model = ' + args.model + '(dataset.get_shape(), args.rank, args.init)')


exec('regularizer = ' + args.regularizer + '(args.reg)')
regularizer = [regularizer, N3(args.reg)]
device = torch.device('cuda')
model.to(device)
for reg in regularizer:
    reg.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

scheduler = None
if args.model == 'MSDCSE':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_method, 'min', factor=args.factor, min_lr=args.min_lr, patience=args.patience
    )
print("Using scheduler:", scheduler)

optimizer = KBCOptimizer(
    datasets, args.dataset, args.model, model, regularizer, optim_method, args.batch_size,
    args.rank, args.out_size, scheduler=scheduler,
)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}, m


cur_loss = 0
base_mrr = 0
test_res = None
if args.checkpoint != '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))


def save_args():
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
    elif len_emb == 3:
        np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')
    return 1


# 训练
if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        best_valid_mrr = 0.0
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e + 1))
            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)

            if (e + 1) % args.valid == 0:
                (valid, valid_mrr), (test, test_mrr) = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test']
                ]
                print("\t VALID: ", valid)
                print("\t TEST: ", test)

                log_file.write("Epoch: {}\n".format(e + 1))
                log_file.write("\t VALID: {}\n".format(valid))
                log_file.write("\t TEST: {}\n".format(test))
                log_file.flush()

                if args.do_save and valid_mrr > best_valid_mrr:
                    best_valid_mrr = valid_mrr
                    save_args()
                    print(f"\t [SAVE] New best valid MRR: {valid_mrr:.4f}")

# 测试
if args.do_test:
    if args.checkpoint != '':
        model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))
    
    (test, test_mrr) = avg_both(*dataset.eval(model, 'test', -1))
    print("Final TEST:", test)
