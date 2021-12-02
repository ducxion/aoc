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
        self.star = star
        self.test = test
        self.day = math.ceil(int(star)/2)
        self.data = Path.cwd() / "inputs" / "day{}.input".format(self.day)
        if self.test == True:
            self.data = Path.cwd() / "tests/day{}.test".format(self.day)

    def day1(self):        
        array = np.loadtxt(self.data)
        if self.star == "1":
            mesh = np.array(np.meshgrid(array, array)).T.reshape(-1,2)            
        else:
            mesh = np.array(np.meshgrid(array, array, array)).T.reshape(-1,3)
        arr = mesh[np.where(mesh.sum(axis=1) == 2020)][0]
        return np.prod(arr)
    
    def day2(self):
        vcount = 0
        array = np.genfromtxt(self.data, dtype="str")
        shift = 1 if self.star == "4" else 0
        for row in array:
            minv = int(row[0].split("-")[0]) - shift
            maxv = int(row[0].split("-")[1]) - shift
            if self.star == "3":                
                count = row[2].count(row[1][0])
                if (count >= minv) & (count <= maxv):
                    vcount += 1                
            else:
                char = row[1][0]
                if ((row[2][minv] == char) & (row[2][maxv] != char)) | (
                    (row[2][minv] != char) & (row[2][maxv] == char)
                ):
                    vcount += 1

        return vcount

    def day3(self):
        
        def get_trees(y,x, array):              
            nfold = np.ceil(x * array.shape[0] / array.shape[1])
            array = np.tile(array, int(nfold))
            tree_count = list()
            yrange = np.arange(0, array.shape[0], y)
            xrange = np.arange(0, (array.shape[0]*x), x)

            for i in range(len(yrange)):
                tree_count.append(array[yrange[i], xrange[i]])
            
            return sum(tree_count)

        data = np.genfromtxt(self.data, dtype="str",comments="%").reshape(-1,1)
        data = np.char.replace(data, ".", "0")
        data = np.char.replace(data, "#", "1")

        # create & fill container
        array = np.zeros((data.shape[0], len(data[0][0])))
        for i in range(data.shape[0]):
            array[i][:] = np.fromiter(data[i][0], dtype=int)

        tree_count = list()
        if self.star == "5":
            yxlist = [(1,3)]
        else:
            yxlist = [(1,1), (1,3), (1,5), (1,7), (2,1)]

        for (y,x) in yxlist:
            tree_count.append(get_trees(y,x, array))

        return np.prod(tree_count)

    def day4(self): 

        df = pd.DataFrame(columns=["byr","iyr", "eyr", "hgt", "hcl", "ecl", "pid", "cid"])
        df = df.append(pd.Series(), ignore_index=True)
        count=0

        with open(self.data) as file:
            for line in file:
                if line == "\n":
                    count+=1
                    df = df.append(pd.Series(), ignore_index=True)
                    continue
                for entry in line.split():
                    col = entry.split(":")[0]
                    val = entry.split(":")[1]
                    df.iloc[count][col] = val

        if self.star == "7":
            df["val"] = True
            df["val"][df[df.columns[:-2]].isna().sum(axis=1) > 0] = False
            return df["val"].sum(axis=0)
        
        else:
            # checking for valid and invalid
            for col in ["byr", "iyr", "eyr"]:
                df[col][df[col].str.len() != 4] = np.nan
            df = df.astype(dtype={"byr":float, "iyr":float, "eyr":float})

            df.byr[(df.byr < 1920) | (df.byr > 2002)] = np.nan  # byr
            df.iyr[(df.iyr < 2010) | (df.iyr > 2020)] = np.nan  # iyr
            df.eyr[(df.eyr < 2020) | (df.eyr > 2030)] = np.nan  # eyr
            # hgt
            df.hgt[~((df.hgt.str[:-2].str.isnumeric()) & (df.hgt.str[-2:].isin(["cm","in"])))] = np.nan
            hgtcm = pd.to_numeric(df.hgt[df.hgt.str[-2:] == "cm"].str[:-2])
            hgtin = pd.to_numeric(df.hgt[df.hgt.str[-2:] == "in"].str[:-2])
            df.hgt.loc[hgtcm[(hgtcm < 150) | (hgtcm > 193)].index] = np.nan
            df.hgt.loc[hgtin[(hgtin < 59) | (hgtin > 76)].index] = np.nan
            # hcl
            df.hcl[~((df.hcl.str[0] == "#") & (df.hcl.str.len() == 7) & (df.hcl.str[1:].str.contains(r'[0-9a-fA-F]', regex=True)))] = np.nan
            # ecl
            df.ecl[~(df.ecl.isin(["amb","blu", "brn", "gry", "grn", "hzl", "oth"]))] = np.nan
            # pid
            df.pid[~((df.pid.str.len() == 9) & (df.pid.str.isnumeric()))] = np.nan
            df["val"] = True
            df["val"][df[df.columns[:-2]].isna().sum(axis=1) > 0] = False

            return df["val"].sum(axis=0)

    def day5(self):

        def get_id(entry, rows, columns):
            rowlist = np.arange(0,rows)
            collist = np.arange(0, columns)
            entry = list(map(int, [x for x in entry]))
            i = 0 
            while len(rowlist) > 1:
                rowlist = np.split(rowlist, 2)[entry[i]] 
                i += 1
            while len(collist) > 1:
                collist = np.split(collist, 2)[entry[i]]
                i += 1
            return rowlist[0] * 8 + collist[0]
        
        rows = 128
        columns = 8
        
        array = np.genfromtxt(self.data, dtype="str") 
        find = ["F", "L", "B", "R"]
        repl = ["0", "0", "1", "1"]
        for i in range(len(repl)):
            array = np.char.replace(array, find[i], repl[i])
        id_list = list()
        for entry in array:
            id_list.append(get_id(entry, rows, columns))
        
        id_list = sorted(id_list)
        true_list = np.arange(id_list[0], id_list[-1]+1, 1)
        if self.star == "9":
            return max(id_list)
        else:
            return list(set(id_list).symmetric_difference(true_list))[0]

    def day6(self):
        linelist = list()
        countlist = list()
        if self.star == "11":
            with open(self.data) as file:
                for line in file:                
                    if line == "\n":
                        countlist.append(len(set.union(*[set(val) for val in linelist])))
                        linelist = list()
                        continue
                    linelist.append(line[:-1])
                countlist.append(len(set.union(*[set(val) for val in linelist]))) #last entry
            return sum(countlist)
        
        else:
            with open(self.data) as file:
                for line in file:                
                    if line == "\n":
                        countlist.append(len(set.intersection(*[set(val) for val in linelist])))
                        linelist = list()
                        continue
                    linelist.append(line[:-1])
                countlist.append(len(set.intersection(*[set(val) for val in linelist])))
            return sum(countlist)

    def day7(self):

        namelist = list()
        numlist = list()
        with open(self.data) as file:
            for line in file:
                line = re.sub(r"\bcontain\b|\bno\b", "", line)                
                line = re.split(r"bags|bag", line[:-2])                
                line = [re.sub("[,| ]", "", entry) for entry in line]
                #parent = line[0]
                names = [re.findall("[a-z]+", string) for string in line]
                names = [item for sublist in names for item in sublist]                
                nums = [re.findall(r'\d+', string) for string in line[1:]]
                nums = [int(item) for sublist in nums for item in sublist]
                nums.insert(0,0)

                namelist.append(names)
                numlist.append(nums)

        length = max(map(len, namelist))
        namearr = np.array([xi+[None]*(length-len(xi)) for xi in namelist])
        numarr = np.array([xi+[np.nan]*(length-len(xi)) for xi in numlist])

        if self.star == "13":
            master = set()
            candidates = set(['shinygold'])            
            while len(candidates) >= 1:
                next_candidates = list()
                for color in candidates:
                    indices = np.where(namearr==color)
                    [next_candidates.append(i) for i in namearr[indices[0][np.where(indices[1] != 0)],0]]
                    master.add(color)
                candidates = set(next_candidates).difference(master)
            return len(master) - 1 # don't count shinygold
        
        else:
            master=dict()
            candidates = set(['shinygold'])
            for color in candidates:
                indices = np.where(namearr==color)
                indices = indices[0][np.where(indices[1] == 0)]
                p
            children = namearr[indices, 1:]

    def day8(self):
        arr = np.genfromtxt(self.data, dtype="str")
        strval = arr[:,0]
        numval = arr[:,1].astype("int")      
        replay=[len(strval)]
        
        def sequence(i,acc, replay, rest=False):  
            acc = [acc]          
            while i not in replay:                
                command = strval[i]
                replay.append(i)
                
                if command == "nop":                    
                    if rest:                       
                        i += numval[i]
                        rest = False
                    else:
                        i +=1
                    acc.append(0)                        
                elif command == "acc":
                    acc.append(numval[i])
                    i += 1
                elif command == "jmp":                    
                    if rest:
                        i += 1
                        rest = False
                    else:
                        i += numval[i]
                    acc.append(0)
                if i == len(strval):
                    #breakpoint()
                    return acc, False

            return acc,replay

        if self.star == "15":
            acc,_ =  sequence(0,0, replay)
            return sum(acc)
        else:            
            acclist,replay = sequence(0,0, replay)
            for n in range(len(replay)):
                ix = -1*n-1
                i = replay[ix]
                replay_short = replay[:ix]
                acc = sum(acclist[:ix])
                #breakpoint()
                newacc, newreplay = sequence(i,acc, replay_short, rest=True)                
                if newreplay == False:
                    return sum(newacc)

    def day9(self):
        array = np.genfromtxt(self.data, dtype="int")
        preamble = 25

        for i in range(preamble,array.shape[0]):
            val = array[i]
            arr = array[i-preamble:i]
            checklist = [sum(x) for x in list(itertools.combinations(arr,2))]

            if val not in checklist:
                if self.star == "17":
                    return val
                else:
                    break

        maxval = val
        for i in range(len(array)):
            j = i
            val = 0
            while val < maxval:
                j += 1
                val = array[i:j].cumsum()[-1]
                if val == maxval:
                    #breakpoint()
                    return np.sum([array[i:j].min(), array[i:j].max()])
    def day10(self):
        array = np.array(np.genfromtxt(self.data, dtype="int"))
        array = np.insert(array, [0,-1],[0, np.max(array)+3])
        array = np.sort(array)
        if self.star == "19":
            return np.sum(np.diff(array) == 1) * np.sum(np.diff(array) == 3)
        else:
            nchoices = list()
            for i in range(array.shape[0]):  
                nchoices.append(0)              
                for j in range(1,4):
                    if array[i]+j in array:
                        nchoices[i] +=1
            breakpoint()

    def day11(self):
        data = np.genfromtxt(self.data, dtype="str",comments="%").reshape(-1,1)
        data = np.char.replace(data, ".", "2")
        data = np.char.replace(data, "L", "0")

        # create & fill container
        array = np.zeros((data.shape[0], len(data[0][0])))
        for i in range(data.shape[0]):
            array[i][:] = np.fromiter(data[i][0], dtype=int)
        array[array==2] = np.nan

        if self.star == "21":
            array = np.pad(array, ((1,1),(1,1)), "constant", constant_values=np.nan) 
            mask = ~np.ma.masked_invalid(array).mask

            def test_func(values):
                return np.nansum(values)

            footprint = np.array([[1,1,1],
                                [1,0,1],
                                [1,1,1]])
            
            #breakpoint()
            old = np.copy(array)
            new = np.copy(array)
            while True:
                sums = ndimage.generic_filter(old, test_func, footprint=footprint)
                np.putmask(sums, ~mask, np.nan)
                new[sums == 0] = 1
                new[sums >= 4] = 0
                if np.all(new[mask] == old[mask]):
                    return np.nansum(new)
                old = np.copy(new)

        else:
            breakpoint()
            for i,j in np.meshgrid(array):
                val2 = array[:i,j]
                val7 = array[i+1:,j]
                val4 = array[i,:j]
                val6 = array[i,j+1:]       

    def reveal_star(self):
        start = timer()
        output = eval("self.day"+str(self.day)+"()")     
        end = timer()
        print("The secret value for unlocking Star # {}, is {}".format(self.star, output))
        print("This calculation took {} seconds".format(end-start))

test=True
star = input("Which star should I calculate?")
Advent = Advent(star, test)    
Advent.reveal_star()
