from multiprocessing import Pool
import numpy as np
import os
rank=4
def mod_n(a, n):
    # n = 1...rank
    res = a
    for i in range(n-1):
        res = res % product_list[rank-1-i]
    res //= product_list[rank-n]
    res *= product_list[n-1]
    return res
def process_chunk(args):
    global product_list
    chunk, start, product_list = args
    col_chunk = np.zeros(shape=2 ** 32,dtype=np.uint8)
    for a, element in enumerate(chunk):
        addr = a+start
        res_addr = 0
        for i in range(rank):
            n = i + 1
            res_addr += mod_n(addr, n)
        col_chunk[res_addr] = element
    return col_chunk
def r2c_transpose(num_process, arr):
    try:
        assert 32 % rank == 0 and rank > 1, "rank should be 2,4,8,16,32"
    except AssertionError as e:
        print(e)
        return 1
    square = pow(2, 32//rank)
    product_list = [pow(square,i) for i in range(rank)]
    pool = Pool(processes=num_process)
    chunk_size = len(arr) // num_process
    arg_list = [(arr[i:i+chunk_size], i, product_list) for i in range(0, len(arr), chunk_size)]
    chunk_results = pool.map(process_chunk, arg_list)
    pool.close()
    pool.join()
    result = chunk_results[0]
    for c in chunk_results[1:]:
        result += c
    return result
if __name__ == "__main__":
    arr = np.random.randint(0,256,size=2**32,dtype=np.uint8)
    shaped_arr = arr.reshape((256,256,256,256))
    #print(shaped_arr)
    answer = shaped_arr.transpose(3,2,1,0).flatten()
    print(answer)
    res = r2c_transpose(24,arr)
    print(res)
    if np.array_equal(answer,res):
        print('YES')

# print(os.cpu_count())