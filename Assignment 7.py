import re, string
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import nltk
for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4',
          'averaged_perceptron_tagger','averaged_perceptron_tagger_eng']:
    nltk.download(p, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, WordNetLemmatizer
from nltk.util import ngrams

print("Name: Madhav Keshav Krish | Roll: 1024170356")

# ── paste generate_dataset() & process_ticket() EXACTLY from the notebook ──
# (kept unchanged as required)
tickets, fp = generate_dataset("1024170356")
print("Fingerprint:", fp, "| tickets:", len(tickets))
STOP = set(stopwords.words('english'))

# ================= Q1  =================
rows = []
for i,t in enumerate(tickets,1):
    r = process_ticket(t)
    rows.append(dict(Ticket=f"T{i}", Sentences=len(r['sentences']),
                     Tokens=len(r['tokens']), Punct=len(r['punct']),
                     Stopwords=len(r['stops']), Unique=len(r['unique']),
                     Final=len(r['final'])))
df = pd.DataFrame(rows).set_index('Ticket')
tot = df.sum()
tot['Unique'] = len(set().union(*[process_ticket(t)['unique'] for t in tickets]))
df.loc['TOTAL'] = tot
print("\nQ1 table:\n", df)

sel = process_ticket(tickets[0])       # sample ticket T1
print("\nT1 tokens      :", sel['tokens'])
print("T1 punctuation :", sel['punct'])
print("T1 stopwords   :", sel['stops'])
print("T1 final       :", sel['final'])

# ================= Q2  =================
TEST_LINES = [
 "Isn't the Wi-Fi down? See https://support.example.com/help or mail admin@corp.co.in.",
 "Version 2.0.1 fixed the 95.5% crash; call +91-98765-43210 now.",
 "The state-of-the-art router won't pair; Ivan didn't reply."]

RE_LONG  = re.compile(r"\b[A-Za-z]{6,}\b")
RE_NUM   = re.compile(r"\b\d+(?:\.\d+)*\b")          # 95.5, 2.0.1 kept whole
RE_CAP   = re.compile(r"\b[A-Z][A-Za-z]*\b")
RE_VOWEL = re.compile(r"\b[AEIOUaeiou][A-Za-z]*\b")
RE_MAIL  = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_URL   = re.compile(r"(?:https?://|www\.)\S+")
RE_PH    = re.compile(r"\+?\d[\d()\-\s]{7,}\d")

for src in (TEST_LINES, tickets[:3]):
    print("\nQ2 extraction:")
    for s in src:
        print("  long :", RE_LONG.findall(s))
        print("  num  :", RE_NUM.findall(s))
        print("  caps :", RE_CAP.findall(s))
        print("  vowel:", RE_VOWEL.findall(s))

def clean(t): return RE_PH.sub("<PHONE>", RE_URL.sub("<URL>", RE_MAIL.sub("<EMAIL>", t)))
print("\nQ2 substitution (TEST_LINES):")
for s in TEST_LINES: print(" ", clean(s))

# ================= Q3  =================
PAT = r"""(?:https?://\S+|www\.\S+)               # URL
       |  [\w.+-]+@[\w-]+\.[\w.-]+              # email
       |  \+?\d[\d()\-\s]{7,}\d                 # phone
       |  \d+(?:\.\d+)+                         # decimal / version
       |  [A-Za-z]+(?:-[A-Za-z]+)+              # Wi-Fi, state-of-the-art
       |  [A-Za-z]+'(?:t|s|re|ve|ll|d|m)\b      # isn't
       |  [A-Za-z]+ | \d+"""
tok = RegexpTokenizer(PAT, flags=re.VERBOSE)
my_tokenize = tok.tokenize

print("\nQ3 my_tokenize vs word_tokenize:")
for s in TEST_LINES + tickets[:2]:
    a, b = my_tokenize(s), word_tokenize(s)
    print("\n ", s)
    print("   mine:", a)
    print("   nltk:", b, "  <-- differs" if a!=b else "")

# ================= Q4  =================
porter, lanc = PorterStemmer(), LancasterStemmer()
rex = RegexpStemmer(r'ing$', min=3)         # suffix chosen from our data
wnl = WordNetLemmatizer()

final_tokens = [w for t in tickets for w in process_ticket(t)['final']]
vocab = sorted({w for w in final_tokens if w.isalpha() and len(w)>1})
tags = nltk.pos_tag(vocab)
pos = {'N':'n','V':'v','J':'a','R':'r'}
lemma = [wnl.lemmatize(w, pos.get(t[0],'n')) for w,t in tags]

diff = [(w, porter.stem(w), lanc.stem(w), rex.stem(w), l)
        for w,l in zip(vocab,lemma)
        if len({porter.stem(w), lanc.stem(w), rex.stem(w), l}) > 1]
print("\nQ4 differing words:\n", pd.DataFrame(diff,
      columns=['Word','Porter','Lancaster','Regexp(ing$)','WordNet+POS']).to_string(index=False))
print("→ 'ing$' chosen because dataset has connecting→connect, crashing→crash.")

# ================= Q5  =================
S1 = [t for tk in tickets for t in process_ticket(tk)['tokens']]
S2 = [t for t in S1 if t not in string.punctuation]
S3 = final_tokens
S4 = [porter.stem(t) for t in S3]

for name,S in [("S1",S1),("S2",S2),("S3",S3),("S4",S4)]:
    print(f"{name}: total={len(S)}  unique={len(set(S))}  TTR={len(set(S))/len(S):.3f}")

print("\nTop10 S1:", Counter(S1).most_common(10))
print("Top10 S3:", Counter(S3).most_common(10))

freq = sorted(Counter(S3).values(), reverse=True)
plt.figure(figsize=(5,3)); plt.loglog(range(1,len(freq)+1), freq, 'o')
plt.xlabel("log(rank)"); plt.ylabel("log(freq)"); plt.title("Q5 Zipf plot – S3")
plt.grid(alpha=.3); plt.show()

print("Top5 bigrams S1:", Counter(ngrams(S1,2)).most_common(5))
print("Top5 bigrams S3:", Counter(ngrams(S3,2)).most_common(5))