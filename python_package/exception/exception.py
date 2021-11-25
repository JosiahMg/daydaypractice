"""
    异常处理机制
    try:
        正常执行代码
    except (err1, err2, ..) as e:
        print(e.args)
        异常执行代码
    finally:
        最后执行的代码
"""

try:
    a = int(input('输入整数 a:'))
    b = int(input('输入整数 b:'))
    print(a/b)

except (ValueError, ArithmeticError) as e:
    print('数据异常', type(e))
    print(e.args)

except Exception:  # Exception父类异常类
    print('其它异常')


# 自定义异常类 raise GenderException

class GenderException(Exception):
    def __init__(self) -> None:
        super().__init__()
        self.errmsg = '性别错误设置异常'

    def __str__(self) -> str:
        return self.errmsg


raise GenderException
