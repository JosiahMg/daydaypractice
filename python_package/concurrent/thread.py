"""
线程: 利用cpu与IO可以同时执行的原理，让cpu在IO执行时也可以执行，从而节省时间
应用场景：IO密集型，如文件操作，网络下载，数据库读取等IO操作
threading.Thread线程的使用方法
函数使用方法:
target: 待执行的函数
args = (): 待执行函数的参数
"""
import threading
import requests
import time


urls = [
    f'https://www.cnblogs.com/#p{page}'
    for page in range(1, 51)
]


def craw(url):
    r = requests.get(url)
    print(url, len(r.text))


# 单线程任务
def single_thread():
    for url in urls:
        craw(url)
    print('single thread end')


# 多线程任务
def multi_thread():
    threads = []
    for url in urls:
        threads.append(threading.Thread(target=craw, args=(url, )))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()  # 主线程需要等待子线程结束才可以结束
    print('multi thread end')


if __name__ == '__main__':
    start = time.time()
    single_thread()
    end = time.time()
    print('single thread cost: ', end-start)

    start = time.time()
    multi_thread()
    end = time.time()
    print('multi thread cost: ', end-start)
