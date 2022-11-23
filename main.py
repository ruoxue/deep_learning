import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf

learning_rate = 0.1
batch_size = 128
num_steps = 3000000
eval_step = 200000

eval_word = ['nine','of' , 'going', 'hardware','britain']

embedding_size = 200
max_vocabulary_size = 5000
min_occurrence = 10
skip_window = 3
num_skips = 2
num_sampled = 64

data_path = 'text8.zip'

with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()
len(text_words)






