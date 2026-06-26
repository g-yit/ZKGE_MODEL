import tqdm
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from models import KBCModel


class KBCOptimizer(object):
    def __init__(
            self, dsl, ds, model_name, model: KBCModel, regularizer: list, optimizer: optim.Optimizer,
            batch_size: int = 256, temp: float = 1.0, rank: int = 2000, out_size: int = 2000,
            verbose: bool = True, scheduler=None, rpcsl_context=None,
            use_rpcsl=False, rpcsl_filter_positives=True,
            rpcsl_strength=1.0, rpcsl_warmup_epochs=0, rpcsl_ramp_epochs=1,
            rpcsl_filtered_only=False,
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
        self.use_rpcsl = use_rpcsl
        self.rpcsl_filter_positives = rpcsl_filter_positives
        self.rpcsl_strength = rpcsl_strength
        self.rpcsl_warmup_epochs = rpcsl_warmup_epochs
        self.rpcsl_ramp_epochs = max(1, rpcsl_ramp_epochs)
        self.rpcsl_filtered_only = rpcsl_filtered_only
        self.rpcsl_context = rpcsl_context
        if self.use_rpcsl:
            if rpcsl_context is None:
                raise ValueError("use_rpcsl=True requires rpcsl_context.")
            self.query_index = rpcsl_context['query_index'].long().cuda()
            self.query_pos_ids = rpcsl_context['query_pos_ids'].long().cuda()
            self.query_pos_mask = rpcsl_context['query_pos_mask'].float().cuda()
            self.rel_epsilon = rpcsl_context['rel_epsilon'].float().cuda()

    def get_rpcsl_scale(self, epoch):
        if epoch < self.rpcsl_warmup_epochs:
            return 0.0
        progress = (epoch - self.rpcsl_warmup_epochs + 1) / float(self.rpcsl_ramp_epochs)
        return min(1.0, max(0.0, progress)) * self.rpcsl_strength

    def rpcsl_loss(self, predictions, input_batch, weight=None, epoch=0):
        truth = input_batch[:, 2]
        heads = input_batch[:, 0]
        rels = input_batch[:, 1]
        batch_size = input_batch.shape[0]

        q_idx = self.query_index[heads, rels]
        valid_query = (q_idx >= 0).float().unsqueeze(1)
        safe_q_idx = torch.clamp(q_idx, min=0)
        pos_ids = self.query_pos_ids[safe_q_idx]
        pos_mask = self.query_pos_mask[safe_q_idx] * valid_query

        truth_col = truth.unsqueeze(1)
        other_pos_mask = pos_mask * (pos_ids != truth_col).float()

        if self.rpcsl_filter_positives:
            filtered_logits = predictions.clone()
            safe_pos_ids = torch.where(other_pos_mask > 0, pos_ids, truth_col.expand_as(pos_ids))
            target_logits = predictions.gather(1, truth_col)
            filtered_logits.scatter_(1, safe_pos_ids, -1e6)
            filtered_logits.scatter_(1, truth_col, target_logits)
            ce_logits = filtered_logits
        else:
            ce_logits = predictions

        ce = F.cross_entropy(ce_logits, truth, reduction='none')

        log_probs = F.log_softmax(predictions, dim=1)
        set_pos_ids = torch.cat([truth_col, pos_ids], dim=1)
        set_pos_mask = torch.cat([torch.ones_like(truth_col, dtype=torch.float), other_pos_mask], dim=1)
        pos_log_probs = log_probs.gather(1, set_pos_ids)
        pos_log_probs = pos_log_probs.masked_fill(set_pos_mask <= 0, -1e9)
        set_loss = -torch.logsumexp(pos_log_probs, dim=1)

        if self.rpcsl_filtered_only:
            eps = torch.zeros_like(self.rel_epsilon[rels])
        else:
            eps = self.rel_epsilon[rels] * self.get_rpcsl_scale(epoch)
        fit = (1.0 - eps) * ce + eps * set_loss

        if weight is not None:
            sample_weight = weight[truth]
            return torch.sum(fit * sample_weight) / torch.sum(sample_weight).clamp_min(1e-9)
        return torch.mean(fit)

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(e)
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        loss_fit_list = []

        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                truth = input_batch[:, 2]

                predictions, factors = self.model.forward(input_batch)

                if self.use_rpcsl:
                    l_fit = self.rpcsl_loss(predictions, input_batch, weight=weight, epoch=e)
                else:
                    l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)

                l = l_fit + l_reg

                loss_fit_list.append(l_fit.detach().cpu().numpy())
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
