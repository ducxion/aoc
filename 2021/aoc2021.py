### ADVENT OF CODE ###
import re
import math
import itertools
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from pathlib import Path
from timeit import default_timer as timer


class Advent:
    def __init__(self, star, test):
        self.star = int(star)
        self.test = test
        self.day = math.ceil(int(star)/2)
        self.data = Path.cwd() / "input" / "day{}.input".format(self.day)
        if self.test == True:
            self.data = Path.cwd() / "test/day{}.test".format(self.day)

    def day1(self):
        array = np.loadtxt(self.data)
        if self.star == 1:            
            return np.sum(np.diff(array) >= 1)
        else:
            return sum(pd.Series(array).rolling(3).sum().diff() >= 1)

    def day2(self):
        array = np.loadtxt(self)
        axis = array[:,0].astype(str)
        val = array[:,1].astype(int)
        
        position = np.sum(val[np.where(axis == 'forward')])
        # depth 
        val[np.where(axis == "up")] *= -1
        val_depth = copy.deepcopy(val)
        val_depth[np.where(axis == "forward")] = 0
        depth = np.cumsum(val_depth)[-1]
        if self.star == 3:
            return depth * position

        else:
            idx = np.where(axis == "forward")
            return np.cumsum(val[idx] * depth[idx])[-1] * position


        if self.star == 3:
            pos_dict = dict()
            for direction in ['forward', 'down', 'up']:
                index = np.where(axis == direction)
                pos_dict[axis] = np.sum(array[index,1].astype(int))            
            return pos_dict['forward'] * (pos_dict['down'] - pos_dict['up'])

        else:
    

        


        
        

    def reveal_star(self):
        start = timer()
        output = eval("self.day"+str(self.day)+"()")     
        end = timer()
        print("The secret value for unlocking Star # {}, is {}".format(self.star, output))
        print("This calculation took {} seconds".format(end-start))

test=False
star = input("Which star should I calculate?")
Advent = Advent(star, test)    
Advent.reveal_star()
