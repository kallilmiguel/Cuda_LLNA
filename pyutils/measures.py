from attr import attrib
import numpy as np
import math

#calculate shannon entropy of a time series x of a network
def shannon_ent(binary_sequence):
    
    shannon = 0

    seq_sum = np.sum(binary_sequence)
    seq_length = len(binary_sequence)
    
    prob_1 = seq_sum/seq_length
    prob_0 = 1 - prob_1

    if(prob_0 >0 and prob_1 >0): 
        shannon -= (prob_0*math.log2(prob_0) + prob_1*math.log2(prob_1))
    elif(prob_0 > 0 and prob_1 <= 0):
        shannon -= prob_0*math.log2(prob_0)
    elif(prob_1 > 0 and prob_0 <=0):
        shannon -= prob_1*math.log2(prob_1)

    return shannon

def shannon_ent_histogram(shannon_array, attributes):
    shannon_histogram = np.zeros(shape = attributes)
    resolution = 1/attributes +0.00001

    for shannon_value in shannon_array:
        index = int(shannon_value//resolution)
        shannon_histogram[index]+=1

    return shannon_histogram

def word_ent(binary_sequence, max_length):
    
    freq = np.zeros(max_length)
    word_size = 0
    counter = 0
    for bit in binary_sequence:
        if(bit == 0):
            if(word_size != 0):
                if(word_size > max_length):
                    word_size = max_length
                freq[word_size - 1] += 1
                word_size = 0
        else:
            word_size +=1
        counter+=1
        if(counter == len(binary_sequence)):
            if(word_size > max_length):
                    word_size = max_length
            freq[word_size - 1] += 1

    return freq

def word_histogram(frequencies, max_length, attributes):

    resolution = max_length // attributes
    histogram = np.zeros(attributes)
    for i in range(frequencies.shape[0]):
        index = int((i)//resolution)
        histogram[index] = np.sum(frequencies[i:i+resolution])

    return histogram


def lempel_ziv_complexity(sequence):
    
    sub_strings = set()
    n = len(sequence)

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)

def lempel_ziv_histogram(lempel_ziv_array, attributes):

    length = len(lempel_ziv_array)
    
    lempel_ziv_array = lempel_ziv_array*math.log(length)/length

    max_value = lempel_ziv_array.max()
    min_value = lempel_ziv_array.min()

    histogram = np.zeros(attributes)
    resolution = (max_value-min_value)/attributes + 0.00001

    for lz_value in lempel_ziv_array:
        index = int((lz_value-min_value)/resolution)
        histogram[index] += 1


    return histogram