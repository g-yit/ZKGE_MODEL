import argparse
import errno
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = ["WN18RR", "FB237", "YAGO3-10", "UMLS", "KINSHIP"]


def _split_path(path, split):
    direct = os.path.join(path, split)
    txt = os.path.join(path, split + ".txt")
    if os.path.exists(direct):
        return direct
    if os.path.exists(txt):
        return txt
    raise FileNotFoundError(f"Cannot find split file for {split} under {path}")


def prepare_dataset(src_path, out_root, name, force=False):
    files = ["train", "valid", "test"]
    out_dir = Path(out_root) / name
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    entities, relations = set(), set()
    for split in files:
        with open(_split_path(src_path, split), "r") as to_read:
            for line in to_read:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)

    entities_to_id = {x: i for i, x in enumerate(sorted(entities))}
    relations_to_id = {x: i for i, x in enumerate(sorted(relations))}
    n_relations = len(relations)
    n_entities = len(entities)
    print(f"{name}: {n_entities} entities and {n_relations} relations")

    for dic, fname in [(entities_to_id, "ent_id"), (relations_to_id, "rel_id")]:
        with open(out_dir / fname, "w") as ff:
            for x, i in dic.items():
                ff.write(f"{x}\t{i}\n")

    for split in files:
        examples = []
        with open(_split_path(src_path, split), "r") as to_read:
            for line in to_read:
                lhs, rel, rhs = line.strip().split("\t")
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
        with open(out_dir / (split + ".pickle"), "wb") as out:
            pickle.dump(np.asarray(examples, dtype="uint64"), out)

    to_skip = {"lhs": defaultdict(set), "rhs": defaultdict(set)}
    for split in files:
        with open(out_dir / (split + ".pickle"), "rb") as inp:
            examples = pickle.load(inp)
        for lhs, rel, rhs in examples:
            to_skip["lhs"][(rhs, rel + n_relations)].add(lhs)
            to_skip["rhs"][(lhs, rel)].add(rhs)

    to_skip_final = {"lhs": {}, "rhs": {}}
    for side, skip in to_skip.items():
        for key, values in skip.items():
            to_skip_final[side][key] = sorted(list(values))
    with open(out_dir / "to_skip.pickle", "wb") as out:
        pickle.dump(to_skip_final, out)

    with open(out_dir / "train.pickle", "rb") as inp:
        train_examples = pickle.load(inp)
    counters = {
        "lhs": np.zeros(n_entities),
        "rhs": np.zeros(n_entities),
        "both": np.zeros(n_entities),
    }
    for lhs, _, rhs in train_examples:
        counters["lhs"][lhs] += 1
        counters["rhs"][rhs] += 1
        counters["both"][lhs] += 1
        counters["both"][rhs] += 1
    for key, value in counters.items():
        counters[key] = value / np.sum(value)
    with open(out_dir / "probas.pickle", "wb") as out:
        pickle.dump(counters, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", default="../src_data")
    parser.add_argument("--out_root", default="../data")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        print("Preparing dataset", dataset)
        try:
            prepare_dataset(
                os.path.join(args.src_root, dataset),
                args.out_root,
                dataset,
                force=args.force,
            )
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                print(f"{dataset} exists, skipping. Use --force to overwrite.")
            else:
                raise


if __name__ == "__main__":
    main()
