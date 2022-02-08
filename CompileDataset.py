# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 15:34:55 2022

@author: Zachary Eckley
"""
#%% Setup
import os
import numpy as np

#look at coordinate files to create master list of airfoil names
Dir = 'D:\\MyWork\\Independent\\HPC Research'
Airfoils = os.listdir(Dir+'\\ModifiedCoordinates')
Airfoils = [s.replace('.txt','') for s in Airfoils]

DataFiles = os.listdir(Dir+'\\OutputPolars\\Complete')

#%% Collect performance data into large array

Dataset = np.zeros([1584,121,201,10])

for F in range(len(Airfoils)):

#just test the first file
#F = 0
    File = open(Dir+'\OutputPolars\\Complete\\'+DataFiles[F])
    FileLines = File.readlines()
    File.close()
    
    #setup indices to pull from text files
    Rows = (np.array([5, 206]).reshape(1,2)).repeat(121,0)
    Scale = ((np.array(range(0,121))*207).reshape(121,1)).repeat(2,1)
    Ranges = Rows+Scale
    Indices = []
    
    for r in range(121):
        start = Ranges[r][0]
        stop = Ranges[r][1]
        Chunk = FileLines[start:stop]
        for a in range(len(Chunk)):
            Line = Chunk[a].split()
            Dataset[F][r][a][0] = Line[1]
            Dataset[F][r][a][1] = Line[2]
            Dataset[F][r][a][2] = Line[3]
            Dataset[F][r][a][3] = Line[4]
            Dataset[F][r][a][4] = Line[5]
            Dataset[F][r][a][5] = Line[6]
            Dataset[F][r][a][6] = Line[7]
            Dataset[F][r][a][7] = Line[8]
            Dataset[F][r][a][8] = Line[9]
            Dataset[F][r][a][9] = Line[10]
    
    print('{:4.0f}\\{}\t\t Data imported for {}'.format(F+1,len(Airfoils),Airfoils[F]))
        
#%% Save the dataset
np.save("Airfoil_Performance_Data",Dataset)

#%% Collect coordinate data in array

Dataset = np.zeros([1584,2,200])

for F in range(len(Airfoils)):

#just test the first file
#F = 0
    File = open(Dir+'\\ModifiedCoordinates\\'+DataFiles[F])
    FileLines = File.readlines()
    File.close()
    
    for c in range(1,201):
        Line = FileLines[c].split()
        Dataset[F, 0, c-1] = Line[0]
        Dataset[F, 1, c-1] = Line[1]
    
    print('{:4.0f}\\{}\t\t Data imported for {}'.format(F+1,len(Airfoils),Airfoils[F]))
    
#%% Save the dataset
np.save(Dir+"\\Airfoil_Coordinate_Data",Dataset)
