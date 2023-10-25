'''
logger.py
The logger module, log everything.
'''

import os
import time

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch.utils.tensorboard.writer import SummaryWriter
from yacs.config import CfgNode


def to_text(path: str):
    """
    To text function, from logger to text
    """

    log_dir = path

    directories = [f.path for f in os.scandir(log_dir) if f.is_dir()]
    directories.append(log_dir)
    info_dict = {}

    for directory in directories:
        # Create an EventAccumulator for the log directory
        if directory == log_dir:
            prefix = None
        else:
            prefix = directory.rstrip("/").split("/")[-1].split("_")[1]
        event_accumulator = EventAccumulator(directory)

        # Initialize the event accumulator
        event_accumulator.Reload()

        # Get a list of available tags (summary types)
        tags = event_accumulator.Tags()['scalars']

        # Access and process data for a specific tag
        for tag in tags:
            if prefix:
                name = f"{prefix} {tag}"
            else:
                name = tag
            data = event_accumulator.Scalars(tag)
            for scalar in data:
                epoch = scalar.step
                value = scalar.value
                if epoch in info_dict:
                    info_dict[epoch].append([name, value])
                else:
                    info_dict.update({epoch: [[name, value]]})

    with open(f"{log_dir}/log.txt", "w", encoding='utf-8') as f:
        for epoch, dic in info_dict.items():
            f.write(f"epoch: {epoch}\n")
            for tag, value in dic:
                f.write(f"{tag}: {value}\n")
            f.write("\n")
        f.flush()


class Logger:
    """
    The logger module, log everything.
    """

    def __init__(self, cfg: CfgNode):
        log_dir = os.path.join(cfg.dataset.dir, cfg.log_dir)
        dataset = cfg.dataset.name
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        task = cfg.dataset.query
        config = cfg.model.name + '_' + str(cfg.seed) + '_' + t
        self.path = os.path.join(log_dir, dataset, task + '_' + config)
        self.file = self.path + '.txt'
        self.writer = SummaryWriter(self.path)
        self.logs = []

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value, step):
        self.writer.add_scalars(tag, value, step)

    def log_histogram(self, tag, values, step, bins='auto'):
        self.writer.add_histogram(tag, values, step, bins)

    def log_image(self, tag, image, step, dataformats='CHW'):
        self.writer.add_image(tag, image, step, dataformats)

    def log(self, log):
        self.logs.append(log)

    def flush(self):
        with open(self.file, '+a', encoding='utf-8') as f:
            for log in self.logs:
                f.write(f'{log}\n')
        self.logs = []

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.flush()
        to_text(self.path)
