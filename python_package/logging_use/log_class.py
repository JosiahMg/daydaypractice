"""
    log日志
    五种打印级别:
    logging.DEBUG
    logging.INFO
    logging.warning
    logging.error
    logging.critical

    format参数中可能用到的格式化串
    %(name)s Logger的名字
    %(levelno)s 数字形式的日志级别
    %(levelname)s 文本形式的日志级别
    %(pathname)s 调用日志输出函数的模块的完整路径名，可能没有
    %(filename)s 调用日志输出函数的模块的文件名
    %(module)s 调用日志输出函数的模块名
    %(funcName)s 调用日志输出函数的函数名
    %(lineno)d 调用日志输出函数的语句所在的代码行
    %(created)f 当前时间，用UNIX标准的表示时间的浮 点数表示
    %(relativeCreated)d 输出日志信息时的，自Logger创建以 来的毫秒数
    %(asctime)s 字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
    %(thread)d 线程ID。可能没有
    %(threadName)s 线程名。可能没有
    %(process)d 进程ID。可能没有
    %(message)s用户输出的消息

"""

import logging

"""
简单的用法
filename: 写入文件
filemode: 写入模式,追加还是覆盖
format: 输出格式
datefmt: 日期格式
%()-8s:
- (): 类型
- -: 左对齐
- 8: 最少8个字符对齐
- s: 字符串
"""
logging.basicConfig(level=logging.DEBUG,
                    filename='demo.log',
                    filemode='w',
                    format='%(asctime)s|%(levelname)8s|%(filename)-8s:%(lineno)s|%(message)-20s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.debug('This is debug log')
logging.info('This is info log')
logging.warning('This is warning log')
logging.error('This is error log')
logging.critical('This is critical log')


"""
    高级用法(推荐)
    记录器(笔)
"""
LOG_NAME = 'applog'

logger = logging.getLogger(LOG_NAME)
logger.setLevel(logging.DEBUG)  # 笔的日志级别高
"""
    处理器(纸)
"""
consoleHander = logging.StreamHandler()  # 打印控制台
consoleHander.setLevel(logging.INFO)

# 不知道打印级别默认使用logger的
fileHander = logging.FileHandler(filename='demo.log', mode='w')
fileHander.setLevel(logging.INFO)

# formatter格式
formatter = logging.Formatter(fmt='%(asctime)s|%(levelname)-8s|%(filename)10s|%(funcName)10s|%(lineno)4s|%(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

# 给处理器设置格式
consoleHander.setFormatter(formatter)
fileHander.setFormatter(formatter)

# 记录器设置写到哪些处理器
logger.addHandler(consoleHander)
logger.addHandler(fileHander)

# 设置过滤器
flt = logging.Filter(LOG_NAME)
# logger.addFilter(flt)   # 给定笔过滤
fileHander.addFilter(flt)   # 给定纸过滤
consoleHander.addFilter(flt)


logger.debug('This is debug log')
logger.info('This is info log')
logger.warning('This is warning log')
logger.error('This is error log')
logger.critical('This is critical log')
