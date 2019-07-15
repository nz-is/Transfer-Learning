# -*- coding: utf-8 -*-
"""hdf5datasetwriter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hkAXzPNuRor8RaYIlewOClA3TjHUDRuV
"""

import h5py 
import os

# This a util class for serializing features extracted by a DNN
# Prep for storing large datas to h5PY into training a Classifier 
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images",
        bufSize=1000):
        print("update")
        if os.path.exists(outputPath):
            raise ValueError("The supplied outputPath \
                             already exists", outputPath)
                             
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, 
                                              dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0 
          
    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        
        #If buffer size exceeds bufSize write/reset the buffer
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
          
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i]= self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i 
        self.buffer = {"data": [], "labels":[]}
          
    #Storing class labels/ Strings for readibility 
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",
                                         (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"])>0:
            self.flush()
          
        self.db.close()