# Q1
import pandas as pd

data = {
    "Tid":[1,2,3,4,5,6,7,8,9,10],
    "Refund":["Yes","No","No","Yes","No","No","Yes","No","No","No"],
    "Marital Status":["Single","Married","Single","Married","Divorced",
                      "Married","Divorced","Single","Married","Single"],
    "Taxable Income":["125K","100K","70K","120K","95K","60K","220K","85K","75K","90K"],
    "Cheat":["No","No","No","No","Yes","No","No","Yes","No","Yes"]
}

df = pd.DataFrame(data)
print(df)


# Q2
print(df.loc[[0,4,7,8]])


# Q3
print(df.loc[3:7])
print(df.iloc[4:9,2:5])
print(df.iloc[:,1:4])


# Q4
df = pd.read_csv("Iris.csv")
print(df.head())


# Q5
df = pd.read_csv("Iris.csv")
df.drop(index=4, inplace=True)
df.drop(df.columns[3], axis=1, inplace=True)
print(df)


# Q6
data = {
    "Employee_ID":[101,102,103,104,105],
    "Name":["Aarav","Rohan","Arjun","Priya","Ananya"],
    "Department":["HR","IT","IT","Marketing","Sales"],
    "Age":[29,34,41,28,38],
    "Salary":[50000,70000,65000,55000,60000],
    "Years_of_Experience":[4,8,10,3,12],
    "Joining_Date":["2020-03-15","2017-07-19","2013-06-01","2021-02-10","2010-11-25"],
    "Gender":["Male","Male","Male","Female","Female"],
    "Bonus":[5000,7000,6000,4500,5000],
    "Rating":[4.5,4.0,3.8,4.7,3.5]
}

df = pd.DataFrame(data)
df.to_csv("employees.csv", index=False)

print(df.shape)
df.info()
print(df.describe())

print(df.head())
print(df.tail(3))

print(df["Salary"].mean())
print(df["Bonus"].sum())
print(df["Age"].min())
print(df["Rating"].max())

df.sort_values("Salary", ascending=False, inplace=True)

def performance(r):
    if r >= 4.5:
        return "Excellent"
    elif r >= 4:
        return "Good"
    return "Average"

df["Performance"] = df["Rating"].apply(performance)

print(df)
print(df.isnull().sum())

df.rename(columns={"Employee_ID":"ID"}, inplace=True)

print(df[df["Years_of_Experience"] > 5])
print(df[df["Department"] == "IT"])

df["Tax"] = df["Salary"] * 0.10
print(df)

df.to_csv("employees_modified.csv", index=False)
