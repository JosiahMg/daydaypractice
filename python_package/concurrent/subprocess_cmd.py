"""
subprocess 可以执行应用程序
args: 要执行的shell命令,或者命令列表
shell: 是否执行命令
cwd:当前的工作目录
env:子进程环境变量
"""

import subprocess


def main():
    # p = subprocess.run(["dir", "/a"], shell=True, capture_output=True, cwd='d:\\')
    p = subprocess.run(args="dir /a", shell=True, capture_output=True)
    if p.returncode == 0:  # 判断返回码
        print('Success')
        print(p.stdout.decode('gb2312'))
    else:
        print('failed')
        print(p.stderr.decode('gb2312'))


if __name__ == '__main__':
    main()
