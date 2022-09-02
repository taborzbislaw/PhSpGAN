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

    f = open('normalizacja.dat','rt')
    lines = f.readlines()
    f.close()

    mean = list(map(float,lines[0].split()))
    std = list(map(float,lines[1].split()))

    mean = np.asarray(mean,dtype = np.float32)
    std = np.asarray(std,dtype = np.float32)


    outputHDFFilePath = 'particles.hd5'
    trainGen = HDF5DatasetGenerator(outputHDFFilePath, 1000)
    gen = trainGen.generator()
    print(trainGen.numParticles)

    print(trainGen.db["particles"].shape)

    print(trainGen.db["particles"][-20:,:]*std + mean)

    #n = 1
    #f = open('real.txt','wt')
    #for i in range(n):
        #if i%10==0:
        #    print(i)
        #dum = next(gen)
        #dum = dum*std +  mean
        #print(dum[:10])
        #print(dum[-10:])
        #dum = dum*std + mean
        #for i in range(dum.shape[0]):
        #    print(dum[i,0],dum[i,1],dum[i,2],dum[i,3],dum[i,4],dum[i,5],dum[i,6],file=f)
        
    #f.close()





