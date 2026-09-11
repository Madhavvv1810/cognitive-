# Q1
L = [1,0,2,4,1,7,0,1,9,7]

print(L)
print([no*10 for no in L])

L += [1,8]
L.insert(3,7)
L.remove(7)
L.pop()
L.sort()

print(L)
print(L[::-1])
print(L[:3])
print(L[-3:])

avg = sum(L)/len(L)
print(avg)
print([no for no in L if no > avg])


# Q2
scores = tuple(L[:8])

print(scores)
print("Highest:", max(scores))
print("Lowest:", min(scores))
print("Reverse:", scores[::-1])

x = int(input("Enter score: "))

if x in scores:
    print("Index:", scores.index(x))
else:
    print("Not found")

a, b, *c = scores
print(a, b, c)


# Q3
import random

random.seed(1024170356)
nums = [random.randint(100,900) for i in range(100)]

even = []
odd = []

for x in nums:
    if x % 2 == 0:
        even.append(x)
    else:
        odd.append(x)

print("Even:", even)
print("Odd:", odd)

def prime(n):
    for i in range(2,n):
        if n % i == 0:
            return False
    return n > 1

p = [x for x in nums if prime(x)]
print("Prime:", p)

count = {}
for x in nums:
    count[x] = count.get(x,0) + 1

most = max(count, key=count.get)
print("Most repeated:", most)
print("Times:", count[most])


# Q4
A = {x*7 for x in [1,2,3,4,5]}
B = {x*9 for x in [1,2,3,4,5]}

print(A)
print(B)
print(A|B)
print(A&B)
print(A-B)
print(B-A)
print(A^B)

print(A <= B)
print(B <= A)

x = int(input("Enter value: "))
A.discard(x)
print(A)


# Q5
d = {
    "name":"Madhav",
    "roll_no":"1024170356",
    "branch":"COPC",
    "age":20,
    "city":"Patiala"
}

print(d)

d["location"] = d.pop("city")
d["cgpa"] = 8.5
d["age"] += 1

print(d)

x = d.copy()
print(x.pop("branch"))
print(x)

x = d.copy()
del x["branch"]
print(x)

for k,v in d.items():
    print(k,v)

print(d.get("email","Email not found"))

friend = {
    "name":"Mantavya",
    "roll_no":"1024030900",
    "branch":"CSE",
    "age":19,
    "city":"chd"
}

print(d | friend)
print({k:v for k,v in d.items() if isinstance(v,str)})
