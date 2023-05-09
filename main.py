from PIL import Image
import numpy as np
from scipy.signal import correlate2d
import pickle 

def sh(i):
    if i<10:
        return [' ', '.', ',', ':', ';', '+', '*', '?', '%', '#', '@'][i]
    else:
        return i

def characterize(test_array):
        params = [0,0,0]
        TX_data = np.sum(test_array, axis=2)
        params[0]+=np.mean(TX_data)
        params[1]+=np.var(TX_data)//(32*6)
        params[2]+= np.max(correlate2d(TX_data, TX_data, mode='same'))//(512*340)
        return [round(i,2) for i in params]


def interp_data(t):
    for i in t:
        for j in i:
            print('{:>3}'.format(f"{sh(round(j)//26)}"), end="")
        print()



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


def obtain_statistics(n, filecount=10722, p=0):
    TX_data = np.zeros((28,28))
    for i in range(filecount):
        im = Image.open(f"dataset\\{n}\\{n}\\{i}.png")
        im_array = np.array(im).astype(int)
        curr_sum = np.sum(im_array, axis=2)
        TX_data = np.add(curr_sum, TX_data)
    return (TX_data/filecount)

gathering_mode=1
fc = 1000
digits=[0,7]

if __name__=="__main__":
    stats = []
    if gathering_mode:
        for i in digits:
            a = obtain_statistics(i, filecount=fc)
            interp_data(a)
            stats.append(a)
            with open("stats.pickle", "wb") as f:
                pickle.dump(stats, f)
    else:
        with open("stats.pickle", "rb") as f:
            stats = pickle.load(f)
            for stat in stats:
                interp_data(stat)

    # Identify Number:
    for n in digits:
        duration =5
        res = 0
        for i in range(duration):
            im =Image.open(f"dataset\\{n}\\{n}\\{i+5312}.png")
            test_arr = np.array(im).astype(int)
            test_arr = (np.sum(test_arr, axis=2))
            
            # Compute the correlation coefficients between the test array and each reference array
            corr_coefs = [round(np.corrcoef(test_arr.flatten(), stat.flatten())[0, 1],2) for stat in stats]
            #print(corr_coefs)

            # Find the index of the reference array with the highest correlation coefficient
            best_match_idx = np.argmax(corr_coefs)
            #print(best_match_idx==n)

            if digits[best_match_idx]==n:
                res+=1

        print(f"{n}: {res/duration}")   







# if __name__=="__main__":
#     stats = []
#     if gathering_mode:
#         for i in range(10):
#             a = (obtain_statistics(i, filecount=1000))
#             print(a)
#             stats.append(a)
#             with open("stats.pickle", "wb") as f:
#                 pickle.dump(stats, f)
#     else:
#         with open("stats.pickle", "rb") as f:
#             stats = pickle.load(f)

#     scol_means = [round(i,2) for i in np.mean(stats, axis=0)]  
#     print(f"Means: {scol_means}")
#     for i in [0,1,2]:
#         for j in range(10):
#             stats[j][i] =round(stats[j][i]-scol_means[i],2)
    
#     for i in range(len(stats)):
#         print(f"{i}: {stats[i]}")

#     # Define the test array
#     print("Correct Detection Rates:")
#     for n in range(10):
#         duration =50
#         res = 0
#         for i in range(duration):
#             im =Image.open(f"dataset\\{n}\\{n}\\{i}.png")
#             test_arr = np.array(im).astype(int)
#             test_arr = characterize(test_arr)
#             test_arr = np.array(test_arr)-np.array(scol_means)
#             #print(f"Test Vector: {test_arr}")
#             # Compute the correlation coefficients between the test array and each reference array
#             corr_coefs = [np.corrcoef(test_arr, stat)[0, 1] for stat in stats]


#             # Find the index of the reference array with the highest correlation coefficient
#             best_match_idx = np.argmax(corr_coefs)
#             #print(best_match_idx==n)

#             if best_match_idx==n:
#                 res+=1

#         print(f"{n}: {res/duration}")