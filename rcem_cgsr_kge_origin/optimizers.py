import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim
from models import KBCModel


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

    def epoch(self, examples: torch.LongTensor, weight=None):
        self.model.train()
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
