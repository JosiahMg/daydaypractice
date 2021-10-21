from smart_open import open


class IterRead:
    def __iter__(self):
        for line in open('https://radimrehurek.com/gensim_3.8.3/auto_examples/core/mycorpus.txt'):
            yield line.strip()


iter_data = IterRead()

for line in iter_data:
    print(line)