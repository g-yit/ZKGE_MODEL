import argparse
import ast
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch import optim

from datasets import Dataset
from models import MSRSCImprove
from optimizers import KBCOptimizer
from regularizers import DURA, DURA_RESCAL, DURA_RESCAL_W, DURA_W, Fro, L1, L2, N3, NA


DATASETS = [
    "WN18RR",
    "FB237",
    "YAGO3-10",
    "UMLS",
    "KINSHIP",
    "WN18",
    "FB15k",
    "FB15k-237",
    "Nations",
]


def parse_list_argument(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}") from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("filter_size_list must be a list.")
    return [tuple(item) for item in parsed]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs["lhs"] + mrrs["rhs"]) / 2.0
    h = (hits["lhs"] + hits["rhs"]) / 2.0
    return {"MRR": m, "hits@[1,3,10]": h}, m


def build_parser():
    parser = argparse.ArgumentParser(description="Improved relation-aware KGE baseline")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--model", type=str, default="MSRSCImprove")
    parser.add_argument("--regularizer", type=str, default="NA")
    parser.add_argument("--optimizer", choices=["Adagrad", "Adam", "SGD"], default="Adam")
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--valid", default=10, type=int)
    parser.add_argument("--rank", default=400, type=int)
    parser.add_argument("--batch_size", default=800, type=int)
    parser.add_argument("--eval_batch_size", default=500, type=int)
    parser.add_argument("--reg", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-3, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--factor", default=0.5, type=float)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("-train", "--do_train", action="store_true")
    parser.add_argument("-test", "--do_test", action="store_true")
    parser.add_argument("-save", "--do_save", action="store_true")
    parser.add_argument("-path", "--save_path", type=str, default="./logs/")
    parser.add_argument("-id", "--model_id", type=str, default="0")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default="")

    parser.add_argument("--input_drop", default=0.3, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    parser.add_argument("--feature_map_drop", default=0.4, type=float)
    parser.add_argument("--output_channel", default=4, type=int)
    parser.add_argument("--k_w", default=20, type=int)
    parser.add_argument("--k_h", default=20, type=int)
    parser.add_argument("--active_fn", default="selu")
    parser.add_argument("--init_fn", default="kaiming_normal")
    parser.add_argument("--filter_size_list", default="[(1,3),(3,3),(1,5)]")

    parser.add_argument(
        "--loss_mode",
        choices=["ce", "soft_ce", "bce"],
        default="soft_ce",
        help="ce is the original single-positive objective; soft_ce is efficient multi-positive CE.",
    )
    parser.add_argument("--max_positives", default=64, type=int)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("-weight", "--do_ce_weight", action="store_true", default=False)

    parser.add_argument("--use_router", dest="use_router", action="store_true")
    parser.add_argument("--no_router", dest="use_router", action="store_false")
    parser.set_defaults(use_router=True)
    parser.add_argument("--router_hidden", default=64, type=int)
    parser.add_argument("--router_temperature", default=1.0, type=float)

    parser.add_argument("--use_anchor", dest="use_anchor", action="store_true")
    parser.add_argument("--no_anchor", dest="use_anchor", action="store_false")
    parser.set_defaults(use_anchor=True)
    parser.add_argument("--anchor_topk", default=8, type=int)
    parser.add_argument("--anchor_alpha", default=0.25, type=float)
    parser.add_argument("--anchor_dropout", default=0.1, type=float)

    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    return parser


def resolve_data_path(path, dataset):
    if os.path.exists(os.path.join(path, dataset)):
        return path
    if os.path.exists(os.path.join("data", dataset)):
        return "data"
    return path


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.filter_size_list = parse_list_argument(args.filter_size_list)
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path(args.data_path, args.dataset)
    dataset = Dataset(data_path, args.dataset)
    examples = torch.from_numpy(dataset.get_train(add_reciprocal=True).astype("int64"))
    relation_stats = dataset.get_relation_stats()
    print("Dataset shape:", dataset.get_shape())
    print(
        "Switches:",
        {
            "loss_mode": args.loss_mode,
            "router": args.use_router,
            "anchor": args.use_anchor,
            "amp": args.use_amp,
        },
    )

    run_dir = None
    if args.do_train or args.do_save:
        save_suffix = (
            f"{args.model}_{args.regularizer}_{args.dataset}_{args.model_id}"
            f"_loss-{args.loss_mode}_router-{int(args.use_router)}_anchor-{int(args.use_anchor)}"
        )
        os.makedirs(args.save_path, exist_ok=True)
        run_dir = os.path.join(args.save_path, save_suffix)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    ce_weight = None
    if args.do_ce_weight and args.loss_mode == "ce":
        ce_weight = torch.as_tensor(dataset.get_weight(), dtype=torch.float32, device=device)

    model = MSRSCImprove(
        num_ent=dataset.get_shape()[0],
        num_rel=dataset.get_shape()[1],
        embedding_dim=args.rank,
        input_drop=args.input_drop,
        hidden_drop=args.hidden_drop,
        feature_map_drop=args.feature_map_drop,
        k_w=args.k_w,
        k_h=args.k_h,
        output_channel=args.output_channel,
        filter_size_list=args.filter_size_list,
        active_fn=args.active_fn,
        init_fn=args.init_fn,
        relation_stats=relation_stats,
        use_router=args.use_router,
        router_hidden=args.router_hidden,
        router_temperature=args.router_temperature,
        use_anchor=args.use_anchor,
        anchor_topk=args.anchor_topk,
        anchor_alpha=args.anchor_alpha,
        anchor_dropout=args.anchor_dropout,
        device=device,
    )
    model.init()
    model.to(device)
    model.set_anchor_resolver(dataset.get_anchor_batch)

    regularizer_map = {
        "NA": NA,
        "N3": N3,
        "Fro": Fro,
        "L1": L1,
        "L2": L2,
        "DURA": DURA,
        "DURA_W": DURA_W,
        "DURA_RESCAL": DURA_RESCAL,
        "DURA_RESCAL_W": DURA_RESCAL_W,
    }
    if args.regularizer not in regularizer_map:
        raise ValueError(f"Unsupported regularizer: {args.regularizer}")
    regularizer = regularizer_map[args.regularizer](args.reg).to(device)

    optim_method = {
        "Adagrad": lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
        "Adam": lambda: optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        ),
        "SGD": lambda: optim.SGD(model.parameters(), lr=args.learning_rate),
    }[args.optimizer]()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_method, "min", factor=args.factor, min_lr=args.min_lr, patience=args.patience
    )

    optimizer = KBCOptimizer(
        dataset=dataset,
        model_name=args.model,
        model=model,
        regularizer=regularizer,
        optimizer=optim_method,
        batch_size=args.batch_size,
        loss_mode=args.loss_mode,
        max_positives=args.max_positives,
        label_smoothing=args.label_smoothing,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        verbose=bool(args.verbose),
        scheduler=scheduler,
    )

    if args.checkpoint:
        ckpt_path = os.path.join(args.checkpoint, "checkpoint")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    def save_model():
        if run_dir is None:
            return
        torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint"))
        np.save(
            os.path.join(run_dir, "entity_embedding.npy"),
            model.emb_ent.weight.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(run_dir, "relation_embedding.npy"),
            model.emb_rel.weight.detach().cpu().numpy(),
        )

    best_valid_mrr = -1.0
    best_saved = False
    if args.do_train:
        log_path = os.path.join(run_dir, "train.log") if run_dir else "train.log"
        with open(log_path, "w") as log_file:
            for epoch in range(args.max_epochs):
                print("Epoch:", epoch + 1)
                train_loss = optimizer.epoch(examples, e=epoch, weight=ce_weight)
                print("\t TRAIN LOSS:", train_loss)

                if (epoch + 1) % args.valid == 0:
                    (valid, valid_mrr), (test, test_mrr) = [
                        avg_both(
                            *dataset.eval(
                                model,
                                split,
                                -1,
                                batch_size=args.eval_batch_size,
                            )
                        )
                        for split in ["valid", "test"]
                    ]
                    print("\t VALID:", valid)
                    print("\t TEST:", test)
                    log_file.write(f"Epoch: {epoch + 1}\n")
                    log_file.write(f"\t TRAIN LOSS: {train_loss}\n")
                    log_file.write(f"\t VALID: {valid}\n")
                    log_file.write(f"\t TEST: {test}\n")
                    log_file.flush()

                    if args.do_save and valid_mrr > best_valid_mrr:
                        best_valid_mrr = valid_mrr
                        best_saved = True
                        save_model()
                        print(f"\t [SAVE] New best valid MRR: {valid_mrr:.4f}")

    if args.do_test:
        if args.checkpoint:
            ckpt_path = os.path.join(args.checkpoint, "checkpoint")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        elif args.do_save and best_saved and run_dir is not None:
            model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint"), map_location=device))

        test, test_mrr = avg_both(
            *dataset.eval(model, "test", -1, batch_size=args.eval_batch_size)
        )
        print("Final TEST:", test)


if __name__ == "__main__":
    main()
