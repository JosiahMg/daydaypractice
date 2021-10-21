from collections import Counter
import nltk

text = """Our legislation on participating in a bid is clear: no one can be prohibited, 
          said Mourao, adding the only thing the company must do is to demonstrate its 
          transparency (in keeping) with the rules that will be established for the process."""


text = nltk.word_tokenize(text)

ct = Counter()


for word in text:
    ct[word] += 1

print(ct)
print(list(ct.elements()))
