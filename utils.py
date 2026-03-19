import logging

from torch.utils.tensorboard.writer import SummaryWriter


class Monitor:
    def __init__(self, max_patience=5, delta=1e-6):
        self.counter_ = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta

    def update(self, cur_value):
        self.counter_ += 1
        is_break = False
        is_lose_patience = False
        if cur_value < self.best_value + self.delta:
            cur_value = 0
            self.patience -= 1
            logging.info("the monitor loses its patience to %d!" %
                         self.patience)
            is_lose_patience = True
            if self.patience == 0:
                self.patience = self.max_patience
                is_break = True
        else:
            self.patience = self.max_patience
            self.best_value = cur_value
            cur_value = 0
        return (is_break, is_lose_patience)

    @property
    def counter(self):
        return self.counter_
