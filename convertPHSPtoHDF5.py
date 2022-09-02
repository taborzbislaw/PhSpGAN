#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math 
import h5py
import os
import glob

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="particles",bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied `outputPath` already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims,dtype="float")
        self.bufSize = bufSize
        self.buffer = {"data": []}
        self.idx = 0

    def add(self, rows):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
            
    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.idx = i
        self.buffer = {"data": []}

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize):
        self.batchSize = batchSize
        self.db = h5py.File(dbPath,'r')
        self.numParticles = self.db["particles"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.numParticles-self.batchSize, self.batchSize):
                particles = self.db["particles"][i: i + self.batchSize]
                yield particles
            epochs += 1

    def close(self):
        self.db.close()



if __name__ == "__main__":

    dataFiles = sorted(glob.glob('/net/scratch/people/plgztabor/primo_workdir/PHSPs_without_VR/ANGLE_*/TXT/Filtered_E*_s*.txt'))

    f = open('normalizacja.dat','rt')
    lines = f.readlines()
    f.close()

    mean = list(map(float,lines[0].split()))
    std = list(map(float,lines[1].split()))

    mean = np.asarray(mean,dtype = np.float32)    # średnie w kolejności E X Y dX dY dZ E_el s_el angle_el
    std = np.asarray(std,dtype = np.float32)

    NumOfParticles = 5760000000
    NumOfFeatures = 9

    outputHDFFilePath = 'particles.hd5'
    writer = HDF5DatasetWriter((NumOfParticles, NumOfFeatures), outputHDFFilePath)

    handles = []
    params1 = []
    params2 = []
    params3 = []
    for dataFile in dataFiles:
        handles.append(open(dataFile,'rt'))
        E = float(os.path.basename(dataFile).split('_')[1][1:4])
        s = float(os.path.basename(dataFile).split('_')[2][1:4])
        angle = float(os.path.dirname(dataFile).split('/')[-2].split('_')[1])
        params1.append(E)
        params2.append(s)
        params3.append(angle)

    print(params3)

    for n in range(NumOfParticles):

        fileId = np.random.randint(0,len(dataFiles))

        features = list(map(float,handles[fileId].readline().split()))[:-1]   # odcinam ostatnią wartość - wagę statystyczną z Primo
        features.append(params1[fileId])
        features.append(params2[fileId])
        features.append(params3[fileId])
        features = np.asarray(features,dtype = np.float32)                    # kolejność z IAEA PHSP: X Y dX dY dZ E E_el s_el

        sign = np.random.randint(0,2)                                         # symetryzacja
        if sign == 0:
            features[0] *= -1
            features[1] *= -1
            features[2] *= -1
            features[3] *= -1

        dum = np.zeros(features.shape,dtype=np.float32)                       # w dum kolejność jak w mean tzn.: E X Y dX dY dZ E_el s_el angle_el
        dum[0] = features[5]
        dum[1:6]= features[0:5]
        dum[6:9] = features[6:9]

        dum = (dum - mean)/std
        
        writer.add([dum])

    writer.close()

    for handle in handles:
        handle.close()

    trainGen = HDF5DatasetGenerator(outputHDFFilePath, 20)
    gen = trainGen.generator()
    print(trainGen.numParticles)

    for n in range(2):
        part = next(gen)
        print(part)





