import re

question = '2000'
question = re.sub('\((\d+/\d+)\)', '\\1', question)

print(question)

print(eval('1/3*3.14*(31.4/(2*3.14))**2*1.2/(10/100*4)'))


1/3*3.14*(31.4/(2*3.14))**2*1.2/(10/100*4)
3.14*25

