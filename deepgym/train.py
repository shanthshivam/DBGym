'''
The config function, get args and cfg
'''
import argparse
from typing import Tuple
from yacs.config import CfgNode

def train(loaders, model, optimizer, scheduler) -> Tuple[argparse.Namespace, CfgNode]:
    '''
    The config function, get args and cfg
    Input: None
    Output: args, cfg
    '''
    return loaders, model, optimizer, scheduler
