import pandas as pd

# Q1

fixed_entries = [
    {
        "question": "what is the annual fee",
        "answer": "The annual fee is Rs 500.",
        "keywords": "fee cost price charge",
        "category": "billing"
    },
    {
        "question": "how to reset password",
        "answer": "Go to Settings > Reset Password.",
        "keywords": "password reset login",
        "category": "account"
    },
    {
        "question": "what are your working hours",
        "answer": "We are open 9 AM to 5 PM.",
        "keywords": "hours timing open time",
        "category": "general"
    },
    {
        "question": "how can i pay the fee",
        "answer": "You can pay via UPI, card, or net banking.",
        "keywords": "pay payment upi fee",
        "category": "billing"
    }
]

d1 = 5
d2 = 6

cat1 = ["billing", "account", "general"][d1 % 3]
cat2 = ["billing", "account", "general"][d2 % 3]

extra_entries = [
    {
        "question": "how can i change my appointment time",
        "answer": "You can change your appointment time from the booking section.",
        "keywords": "appointment change reschedule booking",
        "category": cat1
    },
    {
        "question": "how do i get a receipt for my payment",
        "answer": "Your payment receipt is available after successful payment.",
        "keywords": "receipt payment bill invoice",
        "category": cat2
    }
]

df = pd.DataFrame(fixed_entries + extra_entries)

print("Final DataFrame:")
print(df)


# Q2

def score_query(query, df):
    result = []

    for i in range(len(df)):
        text = df.loc[i, "question"] + " " + df.loc[i, "keywords"]
        words = query.lower().split()

        score = 0

        for word in words:
            if word in text.lower():
                score += 1

        if score > 0:
            result.append({
                "question": df.loc[i, "question"],
                "answer": df.loc[i, "answer"],
                "category": df.loc[i, "category"],
                "score": score
            })

    result = pd.DataFrame(result)

    if len(result) > 0:
        result = result.sort_values("score", ascending=False)

    return result


print("\nQ2:")
print(score_query("fee", df))


# Q3

def same_category(category_name, df):
    return df[df["category"] == category_name]


print("\nQ3:")
print(same_category("general", df))


# Q4

print("\nQ4:")

for i in range(len(df)):
    print(i, df.loc[i, "question"])

choice = int(input("Enter the entry number: "))
keyword = input("Enter a new keyword: ")

df.loc[choice, "keywords"] = df.loc[choice, "keywords"] + " " + keyword

df.to_csv("1024170356_faq_data.csv", index=False)

print("\nUpdated DataFrame:")
print(df)


# Q5

print("\nQ5:")
print(df.groupby("category").size())


# Q6

def best_matches(query, df):
    result = []

    for i in range(len(df)):
        text = df.loc[i, "question"] + " " + df.loc[i, "keywords"]
        words = query.lower().split()

        score = 0

        for word in words:
            if word in text.lower():
                score += 1

        if score > 0:
            result.append({
                "question": df.loc[i, "question"],
                "answer": df.loc[i, "answer"],
                "category": df.loc[i, "category"],
                "score": score
            })

    if len(result) == 0:
        print("No matching entry found.")
        return

    result = pd.DataFrame(result)

    highest = result["score"].max()

    print(result[result["score"] == highest])


print("\nQ6 - Query: fee")
best_matches("fee", df)

print("\nQ6 - Query: xyzabc")
best_matches("xyzabc", df)