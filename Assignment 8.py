
# ============================================================
#                  LAB ASSIGNMENT 8
#              UCS420 - COGNITIVE COMPUTING
#
# Assignment Title: NLP using Python-II
# Feature Extraction, Sentiment Analysis and Text Generation
#
# Name: Madhav Singla
# Enrollment Number: 1024170356
# ============================================================

import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob
from wordcloud import WordCloud

# Downloading required NLTK datasets
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))


# ============================================================
# Q1. TEXT PREPROCESSING AND TOKENIZATION
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 1: TEXT PREPROCESSING AND TOKENIZATION")
print("=" * 55)

# My paragraph about Artificial Intelligence

paragraph = """
Artificial intelligence is one of my favorite technologies.
It makes computers smarter and helps people solve problems.
AI is used in education, healthcare, business and entertainment.
It can save time and improve the quality of human work.
Machine learning allows computers to learn from data.
I believe artificial intelligence will shape our future.
"""

print("\nOriginal Paragraph:")
print(paragraph)

# 1. Convert text to lowercase and remove punctuation

clean_text = re.sub(r'[^\w\s]', '', paragraph.lower())

print("\nLowercase Text Without Punctuation:")
print(clean_text)

# 2. Tokenize into sentences

sentences = sent_tokenize(paragraph)

print("\nSentence Tokenization:")
for sentence in sentences:
    print(sentence)

# 3. Compare split() and word_tokenize()

split_words = paragraph.lower().split()

nltk_words = word_tokenize(paragraph.lower())

print("\nWords using Python split():")
print(split_words)

print("\nWords using NLTK word_tokenize():")
print(nltk_words)

print("\nComparison:")
print("split() separates text based on whitespace.")
print("word_tokenize() uses language rules and separates punctuation.")
print("Therefore, NLTK provides more detailed tokenization.")

# 4. Remove stopwords

tokens = word_tokenize(clean_text)

filtered_words = [
    word for word in tokens
    if word not in stop_words and word.isalpha()
]

print("\nWords After Removing Stopwords:")
print(filtered_words)

# 5. Word Frequency Distribution

frequency = Counter(filtered_words)

print("\nWord Frequency Distribution:")

for word, count in frequency.most_common():
    print(word, ":", count)


# ============================================================
# Q2. STEMMING AND LEMMATIZATION
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 2: STEMMING AND LEMMATIZATION")
print("=" * 55)

# 1. Extract alphabet-only words using re.findall()

alphabet_words = re.findall(
    r'\b[a-zA-Z]+\b',
    paragraph.lower()
)

print("\nAlphabet-only Words:")
print(alphabet_words)

# 2. Remove stopwords

filtered_q2 = [
    word for word in alphabet_words
    if word not in stop_words
]

print("\nAfter Removing Stopwords:")
print(filtered_q2)

# 3. Perform Stemming

stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_q2
]

print("\nStemmed Words:")
print(stemmed_words)

# 4. Perform Lemmatization

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_q2
]

print("\nLemmatized Words:")
print(lemmatized_words)

# 5. Comparison

print("\nComparison Between Stemming and Lemmatization:")

print("Stemming removes word endings to create shorter word forms.")
print("Lemmatization converts words into meaningful dictionary forms.")

print("\nExample:")
print("Stemming of studies:", stemmer.stem("studies"))
print("Lemmatization of studies:", lemmatizer.lemmatize("studies"))

print("\nConclusion:")
print("Stemming is faster and useful in basic text processing.")
print("Lemmatization is preferred when meaningful words are important.")


# ============================================================
# Q3. BAG OF WORDS AND TF-IDF
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 3: BAG OF WORDS AND TF-IDF")
print("=" * 55)

# Three original short texts

texts = [
    "Artificial intelligence improves healthcare and education.",
    "Machine learning helps computers analyze information.",
    "Technology makes communication faster and easier."
]

# 1. Bag of Words using CountVectorizer

count_vectorizer = CountVectorizer()

bow_matrix = count_vectorizer.fit_transform(texts)

print("\nBag of Words Vocabulary:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

# 2. TF-IDF representation

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 3))

# 3. Find top 3 keywords from each text

print("\nTop 3 Keywords from Each Text:")

for i, text in enumerate(texts):

    scores = tfidf_matrix[i].toarray().flatten()

    top_indices = scores.argsort()[-3:][::-1]

    keywords = [
        (feature_names[index], round(scores[index], 3))
        for index in top_indices
    ]

    print("\nText", i + 1, ":", text)
    print("Top 3 Keywords:", keywords)

print("\nInterpretation:")
print("CountVectorizer counts how many times words appear.")
print("TF-IDF gives importance to words in individual documents.")
print("Words with higher TF-IDF scores are more important")
print("in distinguishing one text from the others.")


# ============================================================
# Q4. JACCARD AND COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 4: TEXT SIMILARITY")
print("=" * 55)

# Technology 1: Artificial Intelligence

technology1 = """
Artificial intelligence allows machines to learn from data.
AI helps automate tasks and solve complex problems.
It is widely used in healthcare and education.
AI can improve productivity and save time.
"""

# Technology 2: Blockchain

technology2 = """
Blockchain is a digital technology for secure transactions.
It stores information in connected blocks.
Blockchain improves transparency and data security.
It is useful in banking and digital finance.
"""

# 1. Preprocessing and Tokenization

def preprocess(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return words


words1 = preprocess(technology1)
words2 = preprocess(technology2)

print("\nArtificial Intelligence Tokens:")
print(words1)

print("\nBlockchain Tokens:")
print(words2)

# 2(a). Jaccard Similarity using sets

set1 = set(words1)
set2 = set(words2)

intersection = len(set1.intersection(set2))
union = len(set1.union(set2))

jaccard_similarity = intersection / union if union else 0

print("\nJaccard Similarity:")
print(round(jaccard_similarity, 3))

# 2(b). Cosine Similarity using TF-IDF

tfidf = TfidfVectorizer()

vectors = tfidf.fit_transform([
    technology1,
    technology2
])

cosine_score = cosine_similarity(
    vectors[0],
    vectors[1]
)[0][0]

print("\nCosine Similarity:")
print(round(cosine_score, 3))

# 2(c). Analysis

print("\nAnalysis:")

print("Jaccard similarity measures the overlap between unique words.")

print("Cosine similarity compares the direction of TF-IDF vectors")
print("and considers the importance of words.")

print("In this example, both methods help compare the two technologies.")

print("Cosine similarity is useful for text representations with")
print("different word frequencies and importance.")


# ============================================================
# Q5. SENTIMENT ANALYSIS AND WORD CLOUD
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 5: SENTIMENT ANALYSIS")
print("=" * 55)

# My product reviews

reviews = [
    "This product is amazing and works perfectly.",
    "I really love its quality and performance.",
    "The service was excellent and very helpful.",
    "This product is terrible and completely useless.",
    "I am disappointed with its poor quality.",
    "The service was slow and frustrating.",
    "The product is okay and works normally.",
    "The experience was average."
]

positive_reviews = []

print("\nSentiment Analysis Results:")

# Analyze every review using TextBlob

for review in reviews:

    blob = TextBlob(review)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
        positive_reviews.append(review)

    elif polarity < 0:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    print("\nReview:", review)

    print("Polarity:", round(polarity, 3))

    print("Subjectivity:", round(subjectivity, 3))

    print("Sentiment:", sentiment)


# Word Cloud for positive reviews

print("\nGenerating Word Cloud for Positive Reviews...")

if positive_reviews:

    positive_text = " ".join(positive_reviews)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(positive_text)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title("Word Cloud of Positive Reviews")

    plt.show()

else:
    print("No positive reviews found.")

print("\nSentiment Analysis Completed!")


# ============================================================
# Q6. TEXT GENERATION USING LSTM
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 6: TEXT GENERATION USING LSTM")
print("=" * 55)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Approximately 100 words of original training text

training_text = """
Artificial intelligence is changing the modern world.
Technology helps people communicate and share knowledge.
Machine learning allows computers to learn from experience.
Computers can analyze data and discover useful patterns.
Artificial intelligence is useful in education and healthcare.
Technology makes daily life easier and more productive.
Learning new skills helps students build a better future.
Artificial intelligence continues to improve every year.
Students use technology to explore new ideas and solve problems.
Digital tools make learning more interactive and interesting.
Computers help people complete many tasks efficiently.
The future of technology brings new opportunities.
"""

print("\nOriginal Training Text:")
print(training_text)

# 1. Tokenization using Keras Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts([training_text])

total_words = len(tokenizer.word_index) + 1

print("\nTotal Unique Words:", total_words)

# 2. Create input sequences

sequences = []

for sentence in training_text.split("."):

    sentence = sentence.strip()

    if sentence:

        token_list = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(2, len(token_list) + 1):

            sequences.append(token_list[:i])

# Padding sequences

max_sequence_len = max(len(sequence) for sequence in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

X = sequences[:, :-1]
y = sequences[:, -1]

print("\nInput Sequence Shape:", X.shape)
print("Output Sequence Shape:", y.shape)

# 3. Build a simple LSTM model

model = Sequential([
    Embedding(
        total_words,
        50,
        input_length=max_sequence_len - 1
    ),
    LSTM(100),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nLSTM Model Summary:")

model.summary()

# Train the model

print("\nTraining the LSTM Model...")

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

print("\nModel Training Completed Successfully!")


# Function to generate new text

def generate_text(seed_text, next_words=8):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predicted = model.predict(
            token_list,
            verbose=0
        )

        predicted_word_index = np.argmax(predicted, axis=1)[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_word_index:

                output_word = word
                break

        if not output_word:
            break

        seed_text += " " + output_word

    return seed_text


# Generate 3 new lines using different seed words

print("\nGenerated Text:")

seed_words = [
    "artificial intelligence",
    "technology helps",
    "students use"
]

for seed in seed_words:

    generated = generate_text(seed, 8)

    print("\nSeed:", seed)

    print("Generated Sentence:", generated)


# ============================================================
#                  END OF ASSIGNMENT
# ============================================================

print("\n" + "=" * 55)

print("ALL SIX QUESTIONS COMPLETED SUCCESSFULLY!")

print("Name: Madhav Singla")

print("Enrollment Number: 1024170356")

print("UCS420 - Cognitive Computing")

print("=" * 55)
