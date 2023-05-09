from PIL import Image
import numpy as np
from scipy.signal import correlate2d
import pickle 
import random

def sh(i):
    if i<10:
        return [' ', '.', ',', ':', ';', '+', '*', '?', '%', '#', '@'][i]
    else:
        return i

def interp_data(t):
    for i in t:
        for j in i:
            print('{:>3}'.format(f"{sh(round(j)//32)}"), end="")
        print()
    print(t.sum(axis=1)//32)
    print(t.sum(axis=0)//32)


def extract_sums(t):
    tx = {"row": [], "col": []}
    tx["row"]= t.sum(axis=1)/32
    tx["col"]= t.sum(axis=0)/32
    return tx

def obtain_statistics_old(n, filecount=10772, p=0):
    params = [0,0,0]
    for i in range(filecount):
        im = Image.open(f"dataset\\{n}\\{n}\\{i}.png")
        im_array = np.array(im).astype(int)
        TX_data = np.zeros((28,28))

        for i in range(27):
            for j in range(27):
                pval = sum(im_array[i,j])//32
                TX_data[i,j]=pval
                rowsum = sum(sum(im_array[i]))//32
                if p:
                    print('{:>3}'.format(f"{sh(pval)}"), end="")
            if p:
                print(f"   |   {rowsum}" + "-"*(rowsum//2))

        if p:
            print('-'*28*3)

        for i in range(27):
            colsum = sum(sum(im_array[:,i]))//32
            if p:
                print('{:>3}'.format(f"{colsum}"), end="")

        params[0]+=np.mean(TX_data)
        params[1]+=np.var(TX_data)
        params[2]+= np.max(correlate2d(TX_data, TX_data, mode='same'))//512
    a =([round(i/filecount,2) for i in params])
    return a


def digit_averages(n, filecount=10772):
    tx = {"row": np.zeros(28), "col":np.zeros(28)}
    for i in range(filecount):
        im = Image.open(f"dataset\\{n}\\{n}\\{i}.png")
        im_array = np.array(im).astype(int)[:,:,3]
        tx_i = extract_sums(im_array)
        tx["row"] = np.add(tx["row"], tx_i["row"])
        tx["col"] = np.add(tx["col"], tx_i["col"])
    tx["row"] = tx["row"]/filecount
    tx["col"] = tx["col"]/filecount
    tx["col"] -= np.mean(tx["col"])
    tx["row"] -= np.mean(tx["row"])
    return tx

def hypothesis_test(TX, tx):
    results = {"row": [], "col": []}
    for test_statistic in TX:
        row_corr = max(np.correlate(test_statistic["row"], tx["row"], mode="full"))
        col_corr = max(np.correlate(test_statistic["col"], tx["col"], mode="full"))
        results["row"].append(row_corr)
        results["col"].append(col_corr)
    return results

def compile_data(filecount=10772):
    TX = []
    for n in range(10):
        TX.append(digit_averages(n, filecount=filecount))
    return TX



if __name__=="__main__":
    ### Params ###
    filecount = 1000
    n = 0
    i = random.randint(0,10771)

    ### Execution ###
    TX = compile_data(filecount=filecount)
    im = Image.open(f"dataset\\{n}\\{n}\\{i}.png")
    im_array = np.array(im).astype(int)[:,:,3]
    tx = extract_sums(im_array)
    h = hypothesis_test(TX, tx)

    print(f"Correlations:  <{n},{i}>")
    for i in range(10):
        rowval = '{:>3}'.format(f"{round(h['row'][i]/100)}")
        colval = '{:>3}'.format(f"{round(h['col'][i]/100)}")
        print(f"  {i}: -- Row: {rowval}  Col: {colval}   Cross: {round(round(h['row'][i]/100)*round(h['col'][i]/100) /100)}")






