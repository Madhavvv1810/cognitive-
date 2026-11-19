import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

# Downloading the required NLTK packages
packages = [
    'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for p in packages:
    nltk.download(p, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, WordNetLemmatizer
from nltk.util import ngrams

print("Name: Madhav Keshav Krish | Roll: 1024170356")


# Generating the ticket dataset
# generate_dataset() and process_ticket() are provided in the notebook
tickets, fp = generate_dataset("1024170356")

print("Fingerprint:", fp, "| Total tickets:", len(tickets))

STOP = set(stopwords.words('english'))


# Q1 - Processing tickets and displaying results

rows = []

for i, ticket in enumerate(tickets, 1):
    result = process_ticket(ticket)

    rows.append({
        'Ticket': f"T{i}",
        'Sentences': len(result['sentences']),
        'Tokens': len(result['tokens']),
        'Punct': len(result['punct']),
        'Stopwords': len(result['stops']),
        'Unique': len(result['unique']),
        'Final': len(result['final'])
    })

df = pd.DataFrame(rows).set_index('Ticket')

total = df.sum()
total['Unique'] = len(set().union(
    *[process_ticket(t)['unique'] for t in tickets]
))

df.loc['TOTAL'] = total

print("\nQ1 Ticket summary:\n", df)

# Displaying the details of the first ticket
sample = process_ticket(tickets[0])

print("\nT1 Tokens:", sample['tokens'])
print("T1 Punctuation:", sample['punct'])
print("T1 Stopwords:", sample['stops'])
print("T1 Final tokens:", sample['final'])


# Q2 - Finding different patterns using regular expressions

TEST_LINES = [
    "Isn't the Wi-Fi down? See https://support.example.com/help or mail admin@corp.co.in.",
    "Version 2.0.1 fixed the 95.5% crash; call +91-98765-43210 now.",
    "The state-of-the-art router won't pair; Ivan didn't reply."
]

# Different patterns for extracting text
RE_LONG = re.compile(r"\b[A-Za-z]{6,}\b")
RE_NUM = re.compile(r"\b\d+(?:\.\d+)*\b")
RE_CAP = re.compile(r"\b[A-Z][A-Za-z]*\b")
RE_VOWEL = re.compile(r"\b[AEIOUaeiou][A-Za-z]*\b")
RE_MAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_PH = re.compile(r"\+?\d[\d()\-\s]{7,}\d")

for source in (TEST_LINES, tickets[:3]):
    print("\nQ2 Extracted values:")

    for text in source:
        print("Long words:", RE_LONG.findall(text))
        print("Numbers:", RE_NUM.findall(text))
        print("Capital words:", RE_CAP.findall(text))
        print("Words starting with vowels:", RE_VOWEL.findall(text))


# Replacing phone numbers, URLs and email addresses
def clean(text):
    text = RE_MAIL.sub("<EMAIL>", text)
    text = RE_URL.sub("<URL>", text)
    text = RE_PH.sub("<PHONE>", text)
    return text


print("\nQ2 After replacing details:")

for text in TEST_LINES:
    print(clean(text))


# Q3 - Custom tokenizer and comparison with NLTK

PAT = r"""(?:https?://\S+|www\.\S+)               # URLs
       |  [\w.+-]+@[\w-]+\.[\w.-]+              # Email addresses
       |  \+?\d[\d()\-\s]{7,}\d                 # Phone numbers
       |  \d+(?:\.\d+)+                         # Decimal numbers and versions
       |  [A-Za-z]+(?:-[A-Za-z]+)+              # Hyphenated words
       |  [A-Za-z]+'(?:t|s|re|ve|ll|d|m)\b      # Short forms
       |  [A-Za-z]+ | \d+"""

tokenizer = RegexpTokenizer(PAT, flags=re.VERBOSE)

def my_tokenize(text):
    return tokenizer.tokenize(text)


print("\nQ3 Comparing both tokenizers:")

for text in TEST_LINES + tickets[:2]:
    my_tokens = my_tokenize(text)
    nltk_tokens = word_tokenize(text)

    print("\nText:", text)
    print("My tokenizer:", my_tokens)
    print("NLTK tokenizer:", nltk_tokens)

    if my_tokens != nltk_tokens:
        print("Both tokenizers give different results.")


# Q4 - Stemming and lemmatization

porter = PorterStemmer()
lancaster = LancasterStemmer()
reg_stem = RegexpStemmer(r'ing$', min=3)
lemmatizer = WordNetLemmatizer()

# Collecting all final tokens from the tickets
final_tokens = [
    word
    for ticket in tickets
    for word in process_ticket(ticket)['final']
]

# Getting unique words
vocab = sorted({
    word for word in final_tokens
    if word.isalpha() and len(word) > 1
})

# Finding the POS tags and lemmas
tags = nltk.pos_tag(vocab)

pos_map = {
    'N': 'n',
    'V': 'v',
    'J': 'a',
    'R': 'r'
}

lemmas = [
    lemmatizer.lemmatize(word, pos_map.get(tag[0], 'n'))
    for word, tag in tags
]

# Comparing the results of different methods
differences = []

for word, lemma in zip(vocab, lemmas):
    p = porter.stem(word)
    l = lancaster.stem(word)
    r = reg_stem.stem(word)

    if len({p, l, r, lemma}) > 1:
        differences.append((word, p, l, r, lemma))

result = pd.DataFrame(
    differences,
    columns=['Word', 'Porter', 'Lancaster', 'Regexp(ing$)', 'WordNet+POS']
)

print("\nQ4 Words with different results:\n")
print(result.to_string(index=False))

print("\nThe 'ing$' pattern was selected because words like connecting and crashing occur in the dataset.")


# Q5 - Comparing tokens and word frequencies

S1 = [
    word
    for ticket in tickets
    for word in process_ticket(ticket)['tokens']
]

S2 = [
    word for word in S1
    if word not in string.punctuation
]

S3 = final_tokens

S4 = [porter.stem(word) for word in S3]

# Calculating total words, unique words and TTR
for name, words in [('S1', S1), ('S2', S2), ('S3', S3), ('S4', S4)]:
    total_words = len(words)
    unique_words = len(set(words))
    ttr = unique_words / total_words

    print(
        f"{name}: Total={total_words}, "
        f"Unique={unique_words}, TTR={ttr:.3f}"
    )

# Displaying the most common words
print("\nTop 10 words in S1:", Counter(S1).most_common(10))
print("Top 10 words in S3:", Counter(S3).most_common(10))


# Creating a Zipf plot for S3
freq = sorted(Counter(S3).values(), reverse=True)

plt.figure(figsize=(5, 3))
plt.loglog(range(1, len(freq) + 1), freq, 'o')

plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.title("Q5 Zipf Plot - S3")
plt.grid(alpha=0.3)
plt.show()


# Finding the most common bigrams
print("\nTop 5 bigrams in S1:")
print(Counter(ngrams(S1, 2)).most_common(5))

print("\nTop 5 bigrams in S3:")
print(Counter(ngrams(S3, 2)).most_common(5))
