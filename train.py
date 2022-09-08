# Basic import
import pandas as pd
import numpy as np
import cv2
import math
import os
import re
import matplotlib.pyplot as plt
import sys
import argparse
import shutil
import subprocess
from copy import deepcopy
import logging
import datetime
from pathlib import Path


# Init log file
# Init Date log
_today = datetime.date.today()
_today = _today.strftime("%Y%m%d")
_today_folder = Path(f'./logs/{_today}')
os.makedirs(_today_folder, exist_ok=True)
# Init Exe time log
curr_time = datetime.datetime.now()
_exe_time = curr_time.strftime("%H%M%S")
os.makedirs(_today_folder / _exe_time, exist_ok=True)
log_fn = _today_folder / _exe_time / 'log'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline for ACNE detection')

    parser.add_argument('--data', '-d', default='./data', type=str, help='Input path to data folder - ./data')
    parser.add_argument('--epochs', '-e', default=100, type=int, help='Epochs to train - 100')
    parser.add_argument('--backbone', 
                        default='fasterrcnn_resnet50_fpn', 
                        type=str, 
                        help='Backbone for object detection [fasterrcnn_resnet50_fpn, ] - fasterrcnn_resnet50_fpn')
    parser.add_argument('--resume', default=True, type=bool, help='Resume training model from the last execution - True')
    parser.add_argument('--pretrained', default=True, type=bool, help='Use pretrained weight - True')
    
    parser.add_argument('--optimizer' , '-o', default='adam', type=str, help='Optimizer function [adam, sgd, adamw] - adam')
    parser.add_argument('--learning_rate' , '-lr', default=0.0001, type=float, help='Learning rate - 0.0001')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='Scheduler LR gamma - 0.1')
    parser.add_argument('--lr_step', default=4, type=int, help='Scheduler LR step - 4')
    
    parser.add_argument('--cuda' , '-c', default=0, type=int, help='Use specific GPU - 0')
    parser.add_argument('--resize_image', default=-1, type=int, help='Resize image - (-1: origin size)')
    parser.add_argument('--batch_size','-b', default=4, type=int, help='Batch size - 4')
    parser.add_argument('--num_workers' , '-w', default=4, type=int, help='Use number of workers - 4')
    parser.add_argument('--verbose', '-v', default=10, type=int, help='Log the process and information - 10')
    

    parser.add_argument('--save_epoch', default=10, type=int, help='Number of epoch to save model - 10')
    parser.add_argument('--model_store', default='./models', type=str, help='Place to store model - ./models')
    parser.add_argument('--print_freq', default=100, type=int, help='Number of steps to printout - 100')

    parser.add_argument('--log_level', default='debug', type=str, help='Log Level - Debug')
    
    # Main
    args = parser.parse_args(sys.argv[1:])
    
    # Configure Logging
    ## Log level
    log_level = args.log_level
    if log_level == 'critical':
        log_level = logging.CRITICAL
    elif log_level == 'error':
        log_level = logging.ERROR
    elif log_level == 'warning':
        log_level = logging.WARNING
    elif log_level == 'info':
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    ## Log handler
    log_handlers = [
        logging.FileHandler(filename=log_fn,mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
    ## Log Format
    log_format = '%(asctime)s: %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level,
                        handlers=log_handlers,
                        format=log_format)
    
    logging.info(f'Training log execution {curr_time} stored in {log_fn}')
    

    logging.debug('Argument Parser')
    for k, v in vars(args).items():
        logging.debug('{:20}:    {:}'.format(k, v))

    logging.info('=======================================')
    logging.info('================Prepare================')
    logging.info('=======================================')
    
    import trainer
    trainer.run(args)