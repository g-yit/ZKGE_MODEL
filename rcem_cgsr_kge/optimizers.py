import tqdm
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from models import KBCModel


def multi_positive_softmax_loss(
        logits, positive_ids, positive_mask, class_weight=None
):
    """Uniform-target softmax loss over every known positive for a query.

    Only the positive entity ids are gathered, avoiding a second dense
    ``batch_size x num_entities`` target tensor. Each query contributes equally
    regardless of how many positive tails it has.
    """
    log_probs = F.log_softmax(logits, dim=1)
    positive_log_probs = log_probs.gather(1, positive_ids)
    mask = positive_mask.to(dtype=logits.dtype)

    if class_weight is None:
        positive_count = mask.sum(dim=1).clamp_min(1.0)
        per_query_loss = -(
            positive_log_probs * mask
        ).sum(dim=1) / positive_count
        return per_query_loss.mean()
    else:
        positive_weights = class_weight[positive_ids] * mask
        positive_count = mask.sum(dim=1).clamp_min(1.0)
        weighted_query_loss = -(
            positive_log_probs * positive_weights
        ).sum(dim=1) / positive_count
        mean_query_weight = (
            positive_weights.sum(dim=1) / positive_count
        )
        return weighted_query_loss.sum() / mean_query_weight.sum().clamp_min(
            1e-8
        )


class KBCOptimizer(object):
    def __init__(
            self, dsl, ds, model_name, model: KBCModel, regularizer: list, optimizer: optim.Optimizer,
            batch_size: int = 256, temp: float = 1.0, rank: int = 2000, out_size: int = 2000,
            verbose: bool = True, scheduler=None,
    ):
        self.dsl = dsl
        self.ds = ds
        self.model_name = model_name
        self.scheduler = scheduler
        print(f"Scheduler: {self.scheduler}")
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.temperature = temp
        self.rank = rank
        self.out_size = out_size

    def epoch(self, examples: torch.LongTensor, positive_targets, weight=None):
        self.model.train()
        permutation = torch.randperm(examples.shape[0])
        loss_fit_list = []

        with tqdm.tqdm(total=examples.shape[0], unit='query', disable=not self.verbose) as bar:
            bar.set_description('train multi-positive loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                batch_indices = permutation[
                    b_begin:b_begin + self.batch_size
                ]
                input_batch = examples[batch_indices].clone().cuda()

                target_lists = [
                    positive_targets[i] for i in batch_indices.tolist()
                ]
                max_positive = max(
                    targets.numel() for targets in target_lists
                )
                current_batch_size = len(target_lists)
                positive_ids = torch.zeros(
                    current_batch_size, max_positive,
                    dtype=torch.long, device=input_batch.device,
                )
                positive_mask = torch.zeros(
                    current_batch_size, max_positive,
                    dtype=torch.bool, device=input_batch.device,
                )

                for row, targets in enumerate(target_lists):
                    targets = targets.to(
                        input_batch.device, non_blocking=True
                    )
                    count = targets.numel()
                    positive_ids[row, :count] = targets
                    positive_mask[row, :count] = True

                    # The representative tail is used only by the existing
                    # embedding regularizer. Resample it so every positive tail
                    # can be regularized across epochs.
                    selected = torch.randint(
                        count, (), device=input_batch.device
                    )
                    input_batch[row, 2] = targets[selected]

                predictions, factors = self.model.forward(input_batch)

                l_fit = multi_positive_softmax_loss(
                    predictions,
                    positive_ids,
                    positive_mask,
                    class_weight=weight,
                )
                l_reg = self.regularizer.forward(factors)

                l = l_fit + l_reg

                loss_fit_list.append(l_fit.detach().cpu().item())
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])

                postfix = {'loss': f'{l.item():.2f}'}
                bar.set_postfix(**postfix)

        if self.scheduler is not None:
            self.scheduler.step(np.average(loss_fit_list))
        return l
