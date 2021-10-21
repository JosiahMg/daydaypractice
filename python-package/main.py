from collections import Counter


text = """Our legislation on participating in a bid is clear: no one can be prohibited, 
          said Mourao, adding the only thing the company must do is to demonstrate its 
          transparency (in keeping) with the rules that will be established for the process."""


counter = Counter(text.split())

for k, v in counter.most_common(10):
    print(k)
    print(v)


