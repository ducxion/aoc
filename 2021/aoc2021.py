### ADVENT OF CODE ###
import math
import copy
import itertools
import numpy as np
import pandas as pd
from scipy.stats import mode
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
        array = np.loadtxt(self.data, dtype=str)
        axis = array[:,0].astype(str)
        val = array[:,1].astype(int)
        
        position = np.sum(val[np.where(axis == 'forward')])
        # depth 
        val[np.where(axis == "up")] *= -1
        val_depth = copy.deepcopy(val)
        val_depth[np.where(axis == "forward")] = 0
        depth = np.cumsum(val_depth)
        if self.star == 3:
            return depth[-1] * position

        else:
            idx = np.where(axis == "forward")
            return np.cumsum(val[idx] * depth[idx])[-1] * position

    def day3(self):
        array = np.genfromtxt(self.data, delimiter=1, dtype=int)
        if self.star == 5:
            gamma, epsilon = [],[]
            for col in range(array.shape[1]):
                val = mode(array[:,col])[0][0]
                gamma.append(str(val))
                epsilon.append(str(abs(val-1)))
            gamma = int(''.join(gamma), 2)
            epsilon = int(''.join(epsilon), 2)        
            return gamma * epsilon            

        else:            
            def life_support_loop(array, col, method="O2"):
                column = array[:,col]
                md = mode(column)[0][0]
                count = mode(column)[1][0]
                if method == 'co2':
                    md = abs(md-1)
                if count == len(column)/2: 
                    # case: equal modes when O2
                    md = abs(md-1)
                arr = array[np.where(column == md)]                
                return arr
            
            def o2_co2_loop(array, method):
                arr = copy.deepcopy(array)
                col = 0          
                while arr.shape[0] > 1:
                    arr = life_support_loop(arr, col, method)
                    col += 1                
                b = arr[0]
                return b.dot(2**np.arange(b.size)[::-1])
            
            o2 = o2_co2_loop(array, method='o2')
            co2 = o2_co2_loop(array, method='co2')
            return o2 * co2
    
    def day4(self):
        array = np.genfromtxt(self.data, dtype=int, skip_header=2)
        inputs = np.genfromtxt(self.data, dtype=int, delimiter=',', max_rows=1)        
        zeros = np.zeros_like(array)

        array = np.array(np.split(array, array.shape[0]/5, axis=0))
        zeros = np.array(np.split(zeros, zeros.shape[0]/5, axis=0))
        
        checked = np.array([])
        boxes = np.arange(0, zeros.shape[0])

        for num in inputs:
            zeros[np.where(array == num)] = 1
            for box in np.setdiff1d(boxes, checked):
                if (
                    (np.any(np.sum(zeros[box], axis=0) == 5)) | 
                    (np.any(np.sum(zeros[box], axis=1) == 5))):
                    box_val = np.sum(array[box][np.where(zeros[box] == 0)])
                    if self.star == 7:
                        return box_val * num
                    else:
                        checked = np.append(checked, box)
                        if len(checked) == len(boxes):
                            return box_val * num

    def day5(self):
        inputs = np.genfromtxt(self.data, dtype=str, delimiter=',')
        midcol = np.stack(np.char.split(inputs[:,1], sep=' -> '))
        # columns swapped for x1,x2,y1,y2 format
        array = np.array([inputs[:,0], midcol[:,1], midcol[:,0], inputs[:,-1]], dtype=int).T

        assert sum(sum(array < 0)) == 0 # assumption: no negative values
        xmax = max(np.maximum(array[:,0], array[:,2])) +1
        ymax = max(np.maximum(array[:,1], array[:,3])) +1
        plane = np.zeros((ymax, xmax))

        def draw_line(row, plane):
            coords = {"xx":None, "yy":None} 
            name = ["xx", "yy"]           
            for i in [0,1]:
                if np.diff(row[i]) < 0:
                    coords[name[i]] = np.arange(row[i,0], row[i,1]-1, -1)
                else:
                    coords[name[i]] = np.arange(row[i,0], row[i,1]+1, 1)

            length = max([len(x) for x in coords.values()])
            for name in coords.keys():
                if len(coords[name]) == 1:
                    coords[name] = np.repeat(coords[name], length)            
            
            plane[coords["yy"],coords["xx"]] += 1
            return plane
        
        for row in array:            
            row = row.reshape(2,2)
            if np.any(np.diff(row) == 0):
                plane = draw_line(row, plane)
            elif self.star == 10:
                plane = draw_line(row, plane)
        
        return len(np.where(plane > 1)[0])


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
