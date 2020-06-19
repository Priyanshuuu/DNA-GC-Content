 # Solution 1
 import random
 List_of_tuples = [ ('Book2', '1998', '340'),
                    ('Book3', '2018', '440'),
                    ('Book5',  '1950', '240'),
                    ('Book4',  '1976', '640'),
                    ('Book1',  '2010', '940')]


def sorting_func(sample,term):   
    sample.sort(key = lambda val: val[term]) 
    return sample

by_name, by_year, by_price = 0, 1, 2

sorted_list = sorting_func(List_of_tuples,by_price)

sorted_list = sorting_func(sorted_list,by_name)
print(sorted_list)

def shuf(sample):
    sample = list(sample)
    temp1 = sample[1]
    temp2 = sample[2]
    sample[2] = sample[0]
    sample[1] = temp2
    sample[0] = temp1
    return tuple(sample)

shuffle_list = [ shuf(tup) for tup in sorted_list]
print(shuffle_list)

# Solution 2
def quad(data):
    n = len(A)
    data.sort()
    answer = []  
    for i in range(n - 3): 
        for j in range(i + 1, n - 2): 
            l = j + 1
            r = n - 1
            while (l < r): 
                if(data[i] + data[j] + data[l] + data[r] == X):
                    answer.append([data[i], data[j], data[l], data[r]])
                    l += 1
                    r -= 1
                  
                elif (data[i] + data[j] + data[l] + data[r] < X): 
                    l += 1
                else: 
                    r -= 1
    return answer
    
data = [0, -1, 2, 3, -2, 4, 5]
print(quad(A)) 