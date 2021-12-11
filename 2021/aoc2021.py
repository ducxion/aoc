### ADVENT OF CODE ###
import math
import string
import copy
import itertools
import argparse
import numpy as np
import pandas as pd
from skimage.measure import label
from scipy.stats import mode
from pathlib import Path
from timeit import default_timer as timer


class Advent:
    def __init__(self, args):
        self.star = int(args.star)
        self.test = args.test_input
        self.day = math.ceil(int(self.star)/2)
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
    
    def day6(self):
        array = np.genfromtxt(self.data, dtype=int, delimiter=',')
        days_total = 256 # test: 18 | star: 80
        if self.star == 11:
            time = np.arange(1,days_total+1) 
            for day in time:
                idx = np.where(array == 0)
                array = array - 1
                array[idx] = 6
                array = np.append(array, [8]*len(idx[0]))

            return len(array)

        if self.star == 12:

            def spawn(age, tdays):
                arr = np.zeros(tdays+10, dtype=int)
                arr[age] = 1
                for day in range(tdays):
                    if arr[day] > 0:
                        arr[day+7] += (1*arr[day])
                        arr[day+9] += (1*arr[day])
                return arr[:tdays].sum()

            count = 0
            for age in array:
                count += spawn(age,days_total)
            return count + len(array)
        
    def day7(self):        
        arr = np.genfromtxt(self.data, dtype=int, delimiter=',')
        fuel_counts = []
        for position in np.arange(min(arr), max(arr)+1):
            dup = np.repeat(position, len(arr))
            if self.star == 13:   
                fuel_counts.append(sum( abs(arr-dup)))
            else:
                fuel_counts.append(sum((abs(arr - dup) * ((abs(arr-dup) +1))) / 2)) # n(n+1) / 2
        
        return min(fuel_counts)
    
    def day8(self):
        array = np.genfromtxt(self.data, dtype = str, delimiter = ' ')
        calc_length = np.vectorize(len)
        length_array = calc_length(array)
        if self.star == 15:
            return np.sum(np.isin(length_array[:,11:], [2,3,4,7]))
        else:

            def calculate_row(array, length_array):                
                full = set(string.ascii_lowercase[0:7])
                chardict = dict.fromkeys(list(full))
                numdict = dict.fromkeys(np.arange(0,10))
                lendict = dict.fromkeys(set(length_array[0:10]))
                for i in lendict.keys():
                    lendict[i] = set(array[np.where(length_array == i)])
                for i in range(0,4):
                    nums = [1,4,7,8]
                    lens = [2,4,3,7]
                    numdict[nums[i]] = set(
                        array[np.where(length_array == lens[i])][0])

                def charset(chardict, chars):
                    return set().union(*[chardict[x] for x in list(chars)])

                def char_from_num(number, chardict):
                    legacy = {0:'abcefg', 1:'cf', 2:'acdeg', 3:'acdfg', 4:'bcdf',
                    5:'abdfg', 6:'abdefg', 7:'acf', 8:'abcdefg', 9:'abcdfg'}
                    return set(''.join([''.join(chardict[x]) for x in list(legacy[number])]))

                def num_from_char(char, numdict):
                    return np.where(np.array(list(numdict.values())) == set(char))[0][0]

                # a. 
                chardict['a'] = set(numdict[7]).difference(numdict[1])
                # b.                
                candidates = [full.difference(set(num)) for num in lendict[6]]
                chardict['d'] = numdict[4].difference(
                    numdict[1]).intersection(set().union(*candidates)) 
                chardict['b'] = numdict[4].difference(numdict[1]).difference(chardict['d'])         
                # 5. & f
                numdict[5] = set(np.array(list(lendict[5])) \
                [np.core.defchararray.find(list(lendict[5]), ''.join(chardict['b'])) != -1][0])
                chardict['f'] = numdict[5].intersection(numdict[1])                
                chardict['c'] = numdict[1].difference(chardict['f'])                
                #chardict['b'] = numdict[4].difference(charset(chardict, 'fcd'))                
                chardict['g'] = numdict[5].difference(charset(chardict,'abdf'))                
                chardict['e'] = full.difference(charset(chardict, 'abcdfg'))

                for val in [0,2,3,6,9]:
                    numdict[val] = char_from_num(val, chardict)
                
                output_list = [num_from_char(char,numdict) for char in array[11:]]

                return int(''.join(map(str,output_list))) 
            
            output_integers = list()
            for i in range(0, array.shape[0]):
                output_integers.append(calculate_row(array[i], length_array[i]))
            return sum(output_integers)

    def day9(self):
        array = np.genfromtxt(self.data, dtype=float, delimiter=1)
        if self.star == 17:
            array = np.pad(array, 1, constant_values=9)
            zeros = np.zeros_like(array)
            for i in range(1, array.shape[0]-1):
                for j in range(1, array.shape[1]-1):
                    val = array[i,j] 
                    neighbours = [
                        array[i+1,j],
                        array[i-1,j],
                        array[i,j+1],
                        array[i,j-1]
                        ]
                    if np.all(val < neighbours):
                        zeros[i,j] = array[i,j]+1

            return zeros.sum()

        elif self.star == 18:            
            zeros = np.ones_like(array)
            zeros[np.where(array == 9)] = 0
            regions = label(zeros, connectivity=1)
            counts = [(regions == x).sum() for x in range(1,regions.max()+1)]

            return np.prod(sorted(counts, reverse=True)[0:3])
        
    def day10(self):
        errorlist, scorelist = list(), list()
        mapping = dict(zip(list(')]}>'), list('([{<')))
        errorscore = dict(zip(list(')]}>'), [3,57,1197,25137]))
        comscore = dict(zip(list('([{<'), [1,2,3,4]))

        
        with open(self.data) as file:
            errorlist = list()
            for line in file:
                openlist = list()
                corrupted = False
                for f in list(line):                    
                    if f in list('([{<'):
                        openlist.append(f)       
                    elif f in list(')]}>'):
                        if mapping[f] == openlist[-1]:
                            openlist.pop(-1)
                        else:
                            errorlist.append(f)
                            corrupted = True
                            break
                if not corrupted:
                    score = 0
                    for x in np.flip(openlist):
                        score *= 5
                        score += comscore[x]
                    scorelist.append(score)

        if self.star == 19:
            return sum([errorscore[x] for x in errorlist])
        else:
            return np.median(scorelist)

    def day11(self):

        def flash(xy, array):
            array[xy[0], xy[1]] = np.nan
            xcoords = [xy[0]+i for i in [-1,0,1]]
            ycoords = [xy[1]+i for i in [-1,0,1]]
            coords = np.meshgrid(xcoords, ycoords)
            array[tuple(coords)] += 1
            return array

        array = np.genfromtxt(self.data, dtype=float, delimiter=1)
        array = np.pad(array, 1, constant_values=np.nan)        
        s = slice(1,-1)
        daycount = 0
        flashcount = 0    

        while np.sum(array == 0) != 100:
            array +=1
            daycount += 1
            while np.sum(array > 9) > 0:
                index = np.where(array>9)
                pairs = list(zip(index[0], index[1]))
                for xy in pairs:
                    flashcount += 1
                    array = flash(xy, array)
            array[s,s,][np.isnan(array[s,s])] = 0
            if (self.star == 21) & (daycount == 100):
                return flashcount

        return daycount

    def reveal_star(self):
        start = timer()
        output = eval("self.day"+str(self.day)+"()")     
        end = timer()
        print("The secret value for unlocking Star # {}, is {}".format(self.star, output))
        print("This calculation took {} seconds".format(end-start))

test=False
parser = argparse.ArgumentParser(
    description='A simple Python class called Advent for implementing \
                my Python solutions for Advent of Code 2021 \
                (https://adventofcode.com/2021/about)')

parser.add_argument("-t", "--test_input", help="Runs the code with the test input", action="store_true")
parser.add_argument("-s", "--star", help="Select which star should be calucated", type=int)
args = parser.parse_args()
if not args.star:
    args.star = input("Which star should I calculate?")
Advent = Advent(args)
Advent.reveal_star()
