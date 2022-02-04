# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:39:50 2021

@author: Zachary Eckley
"""

import numpy as np
import matlab
import matlab.engine
import os
#import matplotlib.pyplot as plt

folderName = 'D:\MyWork\Independent\HPC Research\All_UIUC_Airfoils\coord_seligFmt'

Files = os.listdir(folderName)

#open matlab engine
print('Starting Matlab Engine . . . . .')
eng = matlab.engine.start_matlab()
print('Matlab Engine has started\n')

for a in range(len(Files)):
    checkType = str.find(Files[a],'.dat')
    filename = str.replace(Files[a],'.dat','')
    if checkType>-1:
        f = open(folderName + '\\' + Files[a])
        lines = f.readlines()
        Name = lines[0].replace(' AIRFOIL','')
        Name = Name.replace('\n','')
        #Name = Name.split('|')
        #Name = Name[0]
        lines = lines[1:]
        #x = matlab.double(None,[len(lines), 1],False)
        #y = matlab.double(None,[len(lines), 1],False)
        x = np.zeros([len(lines), 1])
        y = np.zeros([len(lines), 1])
        emptyLines = len(lines)
        for L in range(len(lines)):
            if len(lines[L])>1:
                line = lines[L].replace(' 0.',' @0.')
                line = line.replace(' -0.',' @-0.')
                line = line.replace(' .',' @.')
                line = line.replace(' -.',' @-.')
                line = line.replace('\t0.',' @0.')
                line = line.replace('\t-0.',' @-0.')
                points = line.split('@')
                x[L] = float(points[-2])
                y[L] = float(points[-1])
            else:
                emptyLines-=1
                
        x = x[0:emptyLines]
        y = y[0:emptyLines]
        
        #make matlab arrays
        mlX = matlab.double(x.tolist())
        mlY = matlab.double(y.tolist())        

        '''
        #To check if any airfoils shouldnt be used in the set
        fig, axs = plt.subplots()
        axs.plot(x,y)
        axs.set_title(str(Name+'   ('+str(a)+'/1624)'))
        axs.set_aspect('equal')
        axs.set(xlim=(-0.1, 1.1), ylim=(-0.5, 0.5))
        plt.show()
        '''
        
        #run matlab code to apply curve fitting
        ret = eng.SmoothCoords(filename,mlX,mlY)
        print(ret)
        
    #new script will take the coordinates from the txt files and paste into javafoil
    
    #then handle the output data and place that into the same dataframe
    
    #the final dataframe for each airfoil should have the following:
        
        #200 x and y coordinates
        
        #30 reynolds number tests with CL, CD, and CM at several angles of attack
        
    #when the full dictionary has been completed we can begin training the model
    
    
    
# to install the stupid matlab engine install the new version of spyder and reinstall python 

# instead it was as easy as running the command within anaconda prompt