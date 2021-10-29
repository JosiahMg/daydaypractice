import config
import jieba

class JiebaTokenizer:
    def __init__(self):
        self.stopword_path = config.stopword_path
        self.userdict_path = config.userdict_path

    def load_user_dict(self):
        jieba.load_userdict(self.userdict_path)

    def tokeinzier(self, sentense, user_dict=True, stopword=True, lower=True):
        if user_dict:
            self.load_user_dict()
        stopwords = []
        if stopword:
            files = open(self.stopword_path, encoding='utf-8')
            while True:
                line = files.readline()
                line = line.strip()
                if not line:
                    break
                stopwords.append(line)
        return [word for word in jieba.cut(sentense) if stopword and word not in stopwords]


data = 'python常见数据结构有哪些'
print(JiebaTokenizer().tokeinzier(data))