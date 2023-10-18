'''
logger.py
The logging function, log everything
'''

# from yacs.config import CfgNode
from torch.utils.tensorboard.writer import SummaryWriter

# def set_logging(cfg: CfgNode) -> None:
#     '''
#     The seed function, seed everything
#     Input: seed
#     Output: None
#     '''
#     log_dir = "logs"  # Change this to your preferred log directory
#     writer = SummaryWriter(log_dir)
#     cfg = ''
#     print(cfg)

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value, step):
        self.writer.add_scalars(tag, value, step)

    def log_histogram(self, tag, values, step, bins='auto'):
        self.writer.add_histogram(tag, values, step, bins)

    def log_image(self, tag, image, step, dataformats='CHW'):
        self.writer.add_image(tag, image, step, dataformats)

    def close(self):
        self.writer.close()

if __name__ == "__main__":
    # Usage example
    log_dir = "logs"  # Change this to your preferred log directory
    logger = Logger(log_dir)

    for step in range(100):
        loss = 1.0 / (step + 1)
        logger.log_scalar("Loss", loss, step)
        # Replace with your own metrics and data to log

    logger.close()
