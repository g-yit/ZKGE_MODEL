import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim

from models import KBCModel


class KBCOptimizer(object):
    def __init__(
        self,
        dataset,
        model_name,
        model: KBCModel,
        regularizer,
        optimizer: optim.Optimizer,
        batch_size: int = 256,
        loss_mode: str = "soft_ce",
        max_positives: int = 64,
        label_smoothing: float = 0.0,
        use_amp: bool = True,
        grad_clip: float = 0.0,
        verbose: bool = True,
        scheduler=None,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_mode = loss_mode
        self.max_positives = max_positives
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss_fit_list = []

        if weight is not None and self.loss_mode == "ce":
            self.ce_loss = nn.CrossEntropyLoss(reduction="mean", weight=weight)

        with tqdm.tqdm(total=examples.shape[0], unit="ex", disable=not self.verbose) as bar:
            bar.set_description("train loss")
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin : b_begin + self.batch_size].to(self.model.device)
                truth = input_batch[:, 2]
                anchor_ids, anchor_mask = self._make_anchor_batch(input_batch)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions, factors = self.model.forward(
                        input_batch, anchor_ids=anchor_ids, anchor_mask=anchor_mask
                    )
                    l_fit = self._fit_loss(predictions, input_batch, truth)
                    l_reg = self.regularizer.forward(factors)
                    loss = l_fit + l_reg

                self.scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_fit_list.append(float(l_fit.detach().cpu()))
                b_begin += input_batch.shape[0]
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f"{float(loss.detach().cpu()):.3f}")

        avg_loss = float(np.average(loss_fit_list)) if loss_fit_list else 0.0
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
        return avg_loss

    def _make_anchor_batch(self, input_batch):
        if not getattr(self.model, "use_anchor", False):
            return None, None
        return self.dataset.get_anchor_batch(
            input_batch,
            topk=self.model.anchor_topk,
            device=self.model.device,
            exclude_target=True,
        )

    def _fit_loss(self, predictions, input_batch, truth):
        if self.loss_mode == "ce":
            return self.ce_loss(predictions, truth)
        if self.loss_mode == "bce":
            return self._multi_positive_bce(predictions, input_batch)
        if self.loss_mode in ("soft_ce", "multi_ce", "multi_positive"):
            return self._multi_positive_ce(predictions, input_batch)
        raise ValueError("Unknown loss_mode: {}".format(self.loss_mode))

    def _multi_positive_ce(self, predictions, input_batch):
        rows, cols, counts = self.dataset.get_positive_batch(
            input_batch, max_positives=self.max_positives
        )
        device = predictions.device
        rows = rows.to(device)
        cols = cols.to(device)
        counts = counts.to(device)

        log_z = torch.logsumexp(predictions, dim=1)
        selected = predictions[rows, cols]
        pos_sum = torch.zeros(predictions.shape[0], device=device, dtype=predictions.dtype)
        pos_sum.scatter_add_(0, rows, selected)
        pos_mean = pos_sum / counts.clamp_min(1.0)
        loss = log_z - pos_mean

        if self.label_smoothing > 0:
            uniform_loss = log_z - predictions.mean(dim=1)
            eps = self.label_smoothing
            loss = (1.0 - eps) * loss + eps * uniform_loss
        return loss.mean()

    def _multi_positive_bce(self, predictions, input_batch):
        rows, cols, _ = self.dataset.get_positive_batch(
            input_batch, max_positives=self.max_positives
        )
        target = torch.zeros_like(predictions)
        target[rows.to(predictions.device), cols.to(predictions.device)] = 1.0
        if self.label_smoothing > 0:
            target = target * (1.0 - self.label_smoothing) + self.label_smoothing / predictions.shape[1]
        return nn.functional.binary_cross_entropy_with_logits(predictions, target)
