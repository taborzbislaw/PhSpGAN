#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import torch
import random
import pickle
import sys
from libCGAN import *

def normalizacja(filename):

    f = open(filename,'rt')
    lines = f.readlines()
    f.close()

    mean = list(map(float,lines[0].split()))
    std = list(map(float,lines[1].split()))

    mean = np.asarray(mean,dtype = np.float32)
    std = np.asarray(std,dtype = np.float32)

    return mean,std


#############################################################################

json_filename = 'config_001.json'

param_file = open(json_filename).read()
params = json.loads(param_file)

if params['validation_filename'] == 'None':
    params['validation_filename'] = None

if params['start_pth'] == 'None':
    params['start_pth'] = None

params['x_dim'] = len(params['keys'].split())

# normalisation

params['x_mean'],params['x_std'] = normalizacja(params["normalization_data_file"])

# print parameters
for item in params.items():
    print(item)


#check_input_params(params)

# train

print('Building the GAN model ...')
gan = Gan(params)

optim = gan.train()

torch.save(gan.G.state_dict(), 'Gen_' + params['model_name'])
torch.save(gan.D.state_dict(), 'Dis_' + params['model_name'])

with open(params['params_name'], 'wb') as tf:
    pickle.dump(params,tf)





