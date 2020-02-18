from __future__ import division, print_function

import json

import sys
import os
from glob import glob
import re
import wfdb
import cv2
import pandas as pd
import numpy as np
import biosppy
import gc
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import multiprocessing as mp


def signal_to_image(signal, folder_name, record_ind, signal_ind):
    fig = plt.figure(frameon=False)
    plt.plot(signal, linewidth=3.5) 
    plt.xticks([]), plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    filename = folder_name + '/' + str(record_ind) + '_' + str(signal_ind) + '.png'
    
    fig.savefig(filename)
    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, (128, 128))
    cv2.imwrite(filename, im_gray)
    plt.close(fig)

    return im_gray

def get_file_indexes():
    paths = glob('mitdb/*.atr')
    paths = [path[:-4].rsplit("/", 1)[1] for path in paths]
    paths.sort()
    return paths


def create_pic(beats, symbols, signals, record_ind, signal_ind):  
    class_to_idx = {'nor': 1, 'lbb': 2, 'rbb': 5, 'apc': 0, 'pvc': 4, 'pab': 3, 'veb': 6, 'vfw': 7}
    idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

    symbol_to_label = {'N':'nor', 'L':'lbb', 'R':'rbb', 'A':'apc', 'V':'pvc', '/':'pab', 'E':'veb', '!':'vfw'}  
    for i in range(len(beats)):
        if symbols[i] in list(symbol_to_label.keys()):
            left_ind = 0 if i == 0 else beats[i - 1] + 20
            right_ind = len(signals) if i == len(beats) - 1 else beats[i + 1] - 20
            signal = signals[left_ind: right_ind]

            signal_to_image(signal, 'sequential', record_ind, signal_ind)
    
            with open('labels.txt', 'a') as f:
                f.write(str(record_ind) + '_' + str(signal_ind) + ' ' + str(class_to_idx[symbol_to_label[symbols[i]]]))
                f.write('\n')
                
            signal_ind += 1


if __name__ == "__main__":

    class_to_idx = {'nor': 1, 'lbb': 2, 'rbb': 5, 'apc': 0, 'pvc': 4, 'pab': 3, 'veb': 6, 'vfw': 7}
    idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

    symbol_to_label = {'N':'nor', 'L':'lbb', 'R':'rbb', 'A':'apc','V':'pvc', '/':'pab', 'E':'veb', '!':'vfw'}
    
    
    signal_ind = 0

    records = get_file_indexes()

    for record_ind, record in enumerate(records):
        print("Current index: " + str(record_ind))
        signals = wfdb.rdsamp("mitdb/" + record, channels=[0])[0]
        ann = wfdb.rdann("mitdb/" + record, 'atr') 
        symbols = ann.symbol
        beats = list(ann.sample)

        create_pic(beats, symbols, signals, record_ind, signal_ind)

        del(signals)
        del(ann)
        del(symbols)
        del(beats)
        gc.collect()