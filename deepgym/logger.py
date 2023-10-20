'''
logger.py
The logging function, log everything.
'''

import time
import os
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def to_txt(path:str):
    log_directory = path

    subdirectories = [f.path for f in os.scandir(log_directory) if f.is_dir()]

    subdirectories.append(log_directory)
    info_dict = {}

    for path in subdirectories:
        # Create an EventAccumulator for the log directory
        prefix = "" if path == log_directory else path.rstrip("/").split("/")[-1] + "_"
        event_accumulator = EventAccumulator(path)

        # Initialize the event accumulator
        event_accumulator.Reload()

        # Get a list of available tags (summary types)
        tags = event_accumulator.Tags()['scalars']

        # Access and process data for a specific tag
        for tag in tags:
            data = event_accumulator.Scalars(tag)
            print(f"Reading: {prefix + tag}")
            for scalar in data:
                step = scalar.step
                value = scalar.value
                info_dict[step].append([prefix+tag, value]) if step in info_dict else info_dict.update({step: [[prefix+tag, value]]})
    with open(f"{log_directory}/log.txt", "w") as f:
        for step in info_dict:
            f.write(f"Step: {step}\n")
            for tag, value in info_dict[step]:
                f.write(f"Tag: {tag}, Value: {value}\n")
            f.write("\n")
        f.flush()

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.writer = SummaryWriter(log_dir=f'{log_dir}-{current_time}')

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value, step):
        self.writer.add_scalars(tag, value, step)

    def log_histogram(self, tag, values, step, bins='auto'):
        self.writer.add_histogram(tag, values, step, bins)

    def log_image(self, tag, image, step, dataformats='CHW'):
        self.writer.add_image(tag, image, step, dataformats)

    def close(self):
        self.writer.flush()
        self.writer.close()

if __name__ == "__main__":
    to_txt(".")
