import logging
import sys
import os 
import numpy as np 

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def makedir(root, dataset_name, LR, BS, best_test_acc, best_test_fs):
    root = dir + '_runs/'
    sub_name = f'{dataset_name}_lr{LR}_bs{BS}_acc{np.mean(np.array(best_test_acc)):.4f}_fs{np.mean(np.array(best_test_fs)):.4f}'

    os.makedirs(os.path.join(root,sub_name), exist_ok=False)


