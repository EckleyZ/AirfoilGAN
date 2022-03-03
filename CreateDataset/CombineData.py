# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:42:15 2022

@author: Zachary Eckley
"""
import os

# List main directory, subdirectories, and file suffixes
UpperDir = 'D:\MyWork\Independent\HPC Research\OutputPolars'
LowerDir = ['1e6_4e6', '4e6_7e6', '7e6_10e6', '10e6_13e6', '13e6_16e6', '16e6_19e6', '19e6']
suffix = ['_Re_1to4.txt', '_Re_4to7.txt', '_Re_7to10.txt', '_Re_10to13.txt', '_Re_13to16.txt', '_Re_16to19.txt', '_Re_19.txt']

#import filenames by using a subdirectories file list
FileList = os.listdir(UpperDir + '\\' + LowerDir[0])
FileList = [f.replace(suffix[0],'') for f in FileList]

#loop through and combine the files
for Airfoil in FileList:
    CombinedFile = list()
    OutStr = ''
    for R in range(len(LowerDir)):
        File = open(UpperDir + '\\' + LowerDir[R] + '\\' + Airfoil + suffix[R])
        FileLines = File.readlines()
        CombinedFile = CombinedFile + FileLines
        File.close()
        
    CompletedFilename = UpperDir + '\\Complete\\' + Airfoil + '_Complete.txt'
    with open(CompletedFilename, 'w') as f:
        f.write(OutStr.join(CombinedFile))
            
    print('Data combined successfully for ' + Airfoil)
