#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import torch
import random
import pickle
import sys
from matplotlib import pyplot as plt

from libRoCGAN import Generator_RoCGAN,Discriminator,generate_samples2,init_pytorch_cuda,get_min_max_constraints

paramsFileName = 'params.pkl'
modelFileName = 'model.pth'

infile = open(paramsFileName,'rb')
params = pickle.load(infile)
infile.close()

el_energy = float(sys.argv[1])
el_spotSize = float(sys.argv[2])
el_angle = float(sys.argv[3])
nbatches = int(sys.argv[4])

outputFile = 'fake_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_testTime.txt'

params['gpu_mode'] = False
dtypef, device = init_pytorch_cuda(params['gpu_mode'], True)

print(device)

cmin, cmax = get_min_max_constraints(params)
cmin = torch.from_numpy(cmin).type(dtypef)
cmax = torch.from_numpy(cmax).type(dtypef)

loadedGan = Generator_RoCGAN(params,cmin,cmax)
loadedGan.load_state_dict(torch.load('Gen_' + modelFileName,map_location=torch.device(device)))

batch_size = -1
n = 100000
params['current_gpu'] = False

f = open(outputFile,'wt')
for nbatch in range(nbatches):
    if nbatch%50 == 0:
        print(nbatch)
    cond = np.zeros((n,3),dtype=np.float32)
    cond[:,0] = el_energy
    cond[:,1] = el_spotSize
    cond[:,2] = el_angle
    dum = np.asarray(generate_samples2(params, loadedGan, n, batch_size=batch_size, normalize=False,to_numpy=True,cond=cond),dtype=np.float32)
    # w dum in the consecutive columns are '[Ekin X Y dX dY dZ]'
    r = np.random.randint(0,4,size=(dum.shape[0],4))
    #print(r.shape,np.max(r),np.min(r))
    for i in range(dum.shape[0]):     # saving X Y dX dY dZ Ekin, as in IAEA phase spaces
        if r[i,0]==0: 
            print(dum[i,1],dum[i,2],dum[i,3],dum[i,4],dum[i,5],dum[i,0],file=f)
        if r[i,1]==0:
            print(-dum[i,1],-dum[i,2],-dum[i,3],-dum[i,4],dum[i,5],dum[i,0],file=f)
        if r[i,2]==0:
            print(-dum[i,2],dum[i,1],-dum[i,4],dum[i,3],dum[i,5],dum[i,0],file=f)
        if r[i,3]==0:
            print(dum[i,2],-dum[i,1],dum[i,4],-dum[i,3],dum[i,5],dum[i,0],file=f)

f.close() 

