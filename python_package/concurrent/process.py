"""
注意：
from multiprocessing import Process

    多线程时可以理解为重新克隆非代码运行，这个克隆的代码会被全部执行，因此如果不想被执行的必须放在：
    if   __name__  ==  "__main__" :


解决CPU密集型：如判断10000个数是否是素数

下面的程序比较单线程  多线程 以及多进程在计算素数的时间差异

结论： 对于CPU密集型的计算，多线程会可能比单线程更耗时，多进程则时间更短

"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


PRIMES = [112272535095293] * 20


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    sqrt_n = int(math.floor(math.sqrt(n)))

    for i in range(3, sqrt_n+1, 2):
        if n % i == 0:
            return False
    return True


def single_thread():
    for number in PRIMES:
        is_prime(number)


def multi_thread():
    with ThreadPoolExecutor() as pool:
        pool.map(is_prime, PRIMES)


def multi_process():
    with ProcessPoolExecutor() as pool:
        pool.map(is_prime, PRIMES)


if __name__ == '__main__':
    start = time.time()
    single_thread()
    end = time.time()
    print('single_thread, cost: ', end-start, "seconds")

    start = time.time()
    multi_thread()
    end = time.time()
    print('multi_thread, cost: ', end-start, "seconds")

    start = time.time()
    multi_process()
    end = time.time()
    print('multi_process, cost: ', end-start, "seconds")
