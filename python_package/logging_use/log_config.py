"""
功能描述:
使用logging模块的配置文件形式，只做了解

"""
import logging.config


# 加载配置文件
logging.config.fileConfig('log.conf')

# 使用root笔 只记录到console
rootLogger = logging.getLogger()
rootLogger.debug('This is root Logger')

# 使用applog笔 记录到console和file
logger = logging.getLogger('applog')
logger.debug('This is applog Logger')


a = 'abc'
try:
    int(a)
except Exception as e:
    logger.exception(e)
