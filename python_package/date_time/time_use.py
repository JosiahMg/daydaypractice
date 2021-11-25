"""
time模块：
三种类型： 时间戳、结构化时间以及格式化字符串，三者可以相互转换
时间戳：time.time() 从1970.1.1 0:0:0至今
结构化时间: time.struct_time()
格式化字符串: "%Y-%m-%d %H:%M:%S"

时间戳->结构化时间: time.gmtime(stamp: int)
结构化时间->时间戳: time.mktime(st: struct_time)

格式化字符串->结构化时间: strptime()
结构化时间->格式化字符串: strftime()
"""
import time
from pprint import pprint


# 时间戳
t = time.time()
print('1---------------')
pprint(t)

# 结构化时间
# time.struct_time(tm_year=2021, tm_mon=11, tm_mday=17,
# tm_hour=17, tm_min=59, tm_sec=46,
# tm_wday=2, tm_yday=321, tm_isdst=0)
s_t = time.localtime()
print('2---------------')
pprint(s_t)

# 格式化时间
str_time = '2021-07-25 13:13:13'
str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('3---------------')
pprint(str_time)


# 时间戳->结构化时间
st_time = time.gmtime(t)
print('4---------------')
pprint(st_time)

# 结构化时间->时间戳
t = time.mktime(st_time)
print('5---------------')
pprint(t)

# 结构化时间->格式化时间
f_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('6---------------')
pprint(f_time)


# 格式化时间->结构化时间
st_time = time.strptime('2021-07-25 13:13:13', "%Y-%m-%d %H:%M:%S")
print('7---------------')
pprint(st_time)
