"""
正则模块re的使用
"""
import re
from pprint import pprint
from typing import Text, List

content = """\
I rented this one on DVD without any prior knowledge.\
I was suspicious seeing Michael Madsen appearing in a movie I have never heard of, but it was a freebie,\
so why not check it out.<br /><br />Well my guess is that Mr. \
Blonde would very much like to forget he's ever taken part in such a shame of a film.<br /><br />Apparently, \
if your script and dialogs are terrible, even good actors cannot save the day. \
Not to mention the amateur actors that flood this film. \
Too many non-native-English-speakers play parts of native-English-speakers, \
reading out lines from a script that should have been thrown away and not having been made into a movie. \
It's unbelievable how unbelievable all the lines in the movie sound. The music is awful and totally out of place, \
and the whole thing looks and sounds like a poor school play.\
<br /><br />I recommend you watch it just so you would appreciate other, better, movies. \
This is why I gave it a 3 instead of the 1 it deserves.
"""


def tokenize(content: Text) -> List[Text]:
    """
    使用re.sub替换contente中的特定字符，同时按照空间分词
    :param content:
    :return:
    """
    content = re.sub('<.*?>', " ", content)
    filters = ['\.', ',', ':', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&', '\d+']
    content = re.sub('|'.join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


