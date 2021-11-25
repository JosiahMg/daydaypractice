"""
    多线程操作时是共享数据资源，对于共享资源需要进行保护
    lock = threading.Lock()
    
    方法1：
    lock.acquire()
    try:
        # do something
    finally:
        lock.release

    方法2：
    with lock:
        # do something
"""

import threading

lock = threading.Lock()


class Account:
    def __init__(self, balance) -> None:
        self.balance = balance


def draw(account, amount):
    with lock:
        if account.balance >= amount:
            print(threading.current_thread().name, '取钱成功')
            account.balance -= amount
            print(threading.current_thread().name, '余额', account.balance)
        else:
            print(threading.current_thread().name,
                  '取钱失败, 余额不足', account.balance)


if __name__ == '__main__':
    account = Account(1000)
    ta = threading.Thread(name='ta', target=draw, args=(account, 800))
    tb = threading.Thread(name='tb', target=draw, args=(account, 800))

    ta.start()
    tb.start()
