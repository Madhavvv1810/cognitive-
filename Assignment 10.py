
print("Name: Madhav")
print("Roll No: 1024170356")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# Q1. Creating a labelled dataset of healthcare queries
# ============================================================

data = [
    ("What is heart disease?", "heart_basics"),
    ("What does the heart do?", "heart_basics"),
    ("Explain how the heart works.", "heart_basics"),
    ("What is blood pressure?", "heart_basics"),
    ("What is a normal heartbeat?", "heart_basics"),
    ("What are heart valves?", "heart_basics"),
    ("What is cholesterol?", "heart_basics"),
    ("What is coronary artery disease?", "heart_basics"),
    ("What is a heart attack?", "heart_basics"),
    ("What is an irregular heartbeat?", "heart_basics"),

    ("Does smoking affect the heart?", "risk_factors"),
    ("Can obesity cause heart problems?", "risk_factors"),
    ("Is high blood pressure dangerous?", "risk_factors"),
    ("Does diabetes increase heart risk?", "risk_factors"),
    ("Can stress affect heart health?", "risk_factors"),
    ("Does family history affect heart disease?", "risk_factors"),
    ("Is high cholesterol harmful?", "risk_factors"),
    ("Can lack of sleep affect the heart?", "risk_factors"),
    ("Does alcohol affect heart health?", "risk_factors"),
    ("Why is physical inactivity harmful?", "risk_factors"),

    ("How can I keep my heart healthy?", "prevention"),
    ("How can heart disease be prevented?", "prevention"),
    ("Should I check my blood pressure regularly?", "prevention"),
    ("How often should I get a health checkup?", "prevention"),
    ("Can exercise prevent heart disease?", "prevention"),
    ("How can I reduce cholesterol?", "prevention"),
    ("Why should I stop smoking?", "prevention"),
    ("How can I manage stress?", "prevention"),
    ("Is regular walking good for the heart?", "prevention"),
    ("How can I maintain a healthy weight?", "prevention"),

    ("Which foods are good for the heart?", "nutrition"),
    ("Is eating fruits good for heart health?", "nutrition"),
    ("Should I reduce salt intake?", "nutrition"),
    ("Are vegetables good for the heart?", "nutrition"),
    ("Is oily food harmful for the heart?", "nutrition"),
    ("What is a heart healthy diet?", "nutrition"),
    ("Are nuts good for heart health?", "nutrition"),
    ("Should I avoid processed food?", "nutrition"),
    ("Is too much sugar bad for the heart?", "nutrition"),
    ("Why is drinking enough water important?", "nutrition"),

    ("I have severe chest pain and difficulty breathing.", "emergency"),
    ("Someone has collapsed and is not responding.", "emergency"),
    ("I have sudden chest pressure and sweating.", "emergency"),
    ("A person is having severe breathing difficulty.", "emergency"),
    ("There is sudden weakness on one side of the body.", "emergency"),
    ("Someone has fainted and is not waking up.", "emergency"),
    ("I have chest pain spreading to my arm.", "emergency"),
    ("A person is unconscious and gasping for air.", "emergency"),
    ("I am experiencing sudden severe chest discomfort.", "emergency"),
    ("Someone has sudden difficulty speaking and facial drooping.", "emergency")
]

df = pd.DataFrame(data, columns=["Query", "Intent"])

print("\nQ1. Labelled Healthcare Dataset")
print(df.to_string(index=False))

print("\nTotal queries:", len(df))
print("\nQueries in each category:")
print(df["Intent"].value_counts())


# ============================================================
# Q2. Creating a hierarchical taxonomy
# ============================================================

taxonomy = {
    "Heart Health": {
        "Heart Basics": [
            "Heart function",
            "Blood pressure"
        ],
        "Risk Factors": [
            "Smoking",
            "Obesity"
        ],
        "Prevention": [
            "Exercise",
            "Regular checkups"
        ],
        "Nutrition": [
            "Healthy diet",
            "Salt intake"
        ],
        "Emergency": [
            "Chest pain",
            "Breathing difficulty"
        ]
    }
}

print("\nQ2. Healthcare Taxonomy")

for main_category, subcategories in taxonomy.items():
    print("\n", main_category)

    for category, items in subcategories.items():
        print(" ", category, ":", items)


# ============================================================
# Q3. Domain concepts and their relationships
# ============================================================

concepts = [
    ("Heart", "Pumps", "Blood"),
    ("Blood", "Carries", "Oxygen"),
    ("Blood Pressure", "Measures", "Blood Force"),
    ("Cholesterol", "Can contribute to", "Artery Blockage"),
    ("Smoking", "Increases", "Heart Risk"),
    ("Exercise", "Improves", "Heart Fitness"),
    ("Healthy Diet", "Supports", "Heart Health"),
    ("Obesity", "Increases", "Heart Risk"),
    ("Diabetes", "Increases", "Heart Risk"),
    ("Blood Vessels", "Transport", "Blood"),
    ("Salt Intake", "Can increase", "Blood Pressure"),
    ("Stress", "Can affect", "Heart Health")
]

concept_df = pd.DataFrame(
    concepts,
    columns=["Concept 1", "Relationship", "Concept 2"]
)

print("\nQ3. Domain Concepts and Relationships")
print(concept_df.to_string(index=False))


# ============================================================
# Q4. Text preprocessing
# ============================================================

def preprocess(text):

    text = text.lower()

    # Removing punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Splitting text into words
    words = text.split()

    # Removing common stopwords
    stop_words = {
        "is", "the", "a", "an", "are", "and", "or",
        "to", "of", "in", "on", "for", "can", "how",
        "what", "does", "i", "my", "should", "be",
        "it", "do", "have", "has", "why", "me"
    }

    words = [word for word in words if word not in stop_words]

    # Simple suffix removal
    result = []

    for word in words:
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s"):
            word = word[:-1]

        result.append(word)

    return " ".join(result)


df["Processed Query"] = df["Query"].apply(preprocess)

print("\nQ4. Text Preprocessing")
print(df[["Query", "Processed Query"]].head(10).to_string(index=False))


# ============================================================
# Q5. TF-IDF and Naive Bayes classification
# ============================================================

X = df["Processed Query"]
y = df["Intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Converting text into numerical features
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the Naive Bayes model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

# Predicting test data
y_pred = model.predict(X_test_tfidf)

print("\nQ5. Naive Bayes Classification")
print("Model training completed.")


# ============================================================
# Q6. Evaluating the model
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\nQ6. Model Evaluation")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(xticks_rotation=45)
plt.title("Heart Health Assistant - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# Q7. Creating a healthcare knowledge base
# ============================================================

knowledge_base = {

    "heart_basics": {
        "name": "Heart Basics",
        "information": (
            "The heart is a muscular organ that pumps blood "
            "throughout the body. Blood carries oxygen and "
            "nutrients to different organs."
        )
    },

    "risk_factors": {
        "name": "Risk Factors",
        "information": (
            "Common risk factors include smoking, obesity, "
            "high blood pressure, high cholesterol, diabetes, "
            "stress and lack of physical activity."
        )
    },

    "prevention": {
        "name": "Prevention",
        "information": (
            "Heart health can be supported through regular "
            "physical activity, healthy eating, avoiding smoking, "
            "managing stress and regular medical checkups."
        )
    },

    "nutrition": {
        "name": "Nutrition",
        "information": (
            "A balanced diet containing fruits, vegetables, "
            "whole grains and nuts can support heart health. "
            "Limit excess salt, sugar and unhealthy fats."
        )
    },

    "emergency": {
        "name": "Emergency",
        "information": (
            "Possible heart attack symptoms include chest pain, "
            "breathlessness, sweating and pain spreading to the "
            "arm or jaw. Seek emergency medical help immediately."
        )
    }
}


print("\nQ7. Healthcare Knowledge Base")

for key, value in knowledge_base.items():
    print("\n", value["name"])
    print(value["information"])


# ============================================================
# Q8. Reasoning rules
# ============================================================

def apply_rules(query):

    query = query.lower()

    # Rule 1: Emergency symptoms
    emergency_words = [
        "chest pain", "chest pressure", "breathing difficulty",
        "difficulty breathing", "unconscious", "collapsed",
        "not responding", "not waking", "fainted",
        "gasping", "sudden weakness", "facial drooping",
        "difficulty speaking"
    ]

    if any(word in query for word in emergency_words):
        return (
            "EMERGENCY ALERT: These symptoms may indicate "
            "a serious medical emergency. Call your local "
            "emergency number immediately. In India, call 112. "
            "Do not wait for an online assistant."
        )

    # Rule 2: Smoking
    if "smoking" in query:
        return (
            "Smoking increases the risk of heart disease. "
            "Avoiding tobacco can improve heart health."
        )

    # Rule 3: Exercise
    if "exercise" in query or "walking" in query:
        return (
            "Regular physical activity can improve heart fitness. "
            "Choose an activity suitable for your health and "
            "consult a doctor before starting if needed."
        )

    # Rule 4: Salt
    if "salt" in query:
        return (
            "Excessive salt intake can increase blood pressure. "
            "Try to limit high-sodium foods."
        )

    # Rule 5: Healthy diet
    if "diet" in query or "food" in query:
        return (
            "A balanced diet with fruits, vegetables and whole "
            "grains can support heart health."
        )

    return None


print("\nQ8. Reasoning Rules")

rule_examples = [
    "Does smoking affect the heart?",
    "Is exercise good for the heart?",
    "Should I reduce salt?",
    "Which food is good for the heart?",
    "I have severe chest pain."
]

for question in rule_examples:
    print("\nQuestion:", question)
    print("Answer:", apply_rules(question))


# ============================================================
# Q9. Integrating NLP, classification and knowledge retrieval
# ============================================================

def get_response(query):

    # Checking emergency and special rules first
    rule_response = apply_rules(query)

    if rule_response:
        return rule_response

    # Processing the user query
    processed = preprocess(query)

    # Handling empty queries
    if not processed.strip():
        return "Please enter a valid question."

    # Converting query into TF-IDF features
    query_vector = vectorizer.transform([processed])

    # Predicting intent and confidence
    probabilities = model.predict_proba(query_vector)[0]

    best_index = np.argmax(probabilities)

    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    # Handling unsupported or low-confidence queries
    if confidence < 0.25:
        return (
            "Sorry, I could not understand your question. "
            "Please ask something related to heart health."
        )

    # Retrieving information from knowledge base
    response = knowledge_base[intent]["information"]

    return (
        "Detected Intent: " + intent +
        "\nConfidence: " + str(round(confidence * 100, 2)) + "%" +
        "\n\nAssistant: " + response
    )


# ============================================================
# Q10. Interactive Cognitive Assistant
# ============================================================

def run_assistant():

    print("\n====================================")
    print("      HEART HEALTH ASSISTANT")
    print("====================================")

    print("Ask questions about heart health.")
    print("Type 'exit' to close the assistant.")

    while True:

        query = input("\nYou: ")

        if query.lower().strip() == "exit":
            print("Assistant: Thank you for using the assistant.")
            break

        answer = get_response(query)

        print("\nAssistant:", answer)


# ============================================================
# Testing the assistant
# ============================================================

test_queries = [
    "What does the heart do?",
    "Does smoking affect heart health?",
    "How can I prevent heart disease?",
    "Which foods are good for the heart?",
    "I have severe chest pain and difficulty breathing.",
    "Tell me about computer programming."
]

print("\nTesting the Assistant")

for question in test_queries:
    print("\nUser:", question)
    print("Assistant:", get_response(question))


# Starting the interactive assistant
run_assistant()
