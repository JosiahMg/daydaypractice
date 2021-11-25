"""
datatime
"""

import datetime
import time
from pprint import pprint


d = datetime.datetime.today()
pprint(d)  # datetime.datetime(2021, 11, 17, 20, 34, 18, 833701)

d = datetime.datetime.fromtimestamp(time.time())
pprint(d)  # datetime.datetime(2021, 11, 17, 20, 34, 18, 833701)

d = datetime.datetime.strptime('2011年10月10日10点10分10秒', '%Y年%m月%d日%H点%M分%S秒')
pprint(d)

d = datetime.datetime(2011, 10, 12, 8, 4, 15, 900000)
pprint(d)


print(d.year)
print(d.month)
print(d.day)
print(d.hour)
print(d.minute)
print(d.second)
print(d.microsecond)


# 时间差

data_delta = datetime.timedelta(days=10, hours=2)
pprint(data_delta.total_seconds())

d1 = datetime.datetime.today()
d2 = d1 - data_delta
pprint(d2)
pprint(d2.strftime('%Y-%m-%d %H:%M:%S'))
