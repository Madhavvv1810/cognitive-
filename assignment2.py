# Q1
L = [1,0,2,4,1,7,0,1,9,7]

print(L)
print([x*10 for x in L])

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
print([x for x in L if x > avg])


# Q2
scores = tuple(L[:8])

print(scores)
print(max(scores), scores.index(max(scores)))
print(min(scores), scores.index(min(scores)))
print(scores[::-1])

x = int(input("Enter score: "))
print(scores.index(x) if x in scores else "Not found")

a,b,*c = scores
print(a,b,c)


# Q3
import random

random.seed(1024170356)
nums = [random.randint(100,900) for i in range(100)]

even = [x for x in nums if x%2 == 0]
odd = [x for x in nums if x%2 != 0]

print(even)
print(odd)

def prime(n):
    return n > 1 and all(n%i for i in range(2,n))

p = [x for x in nums if prime(x)]
print(p)

count = {x:nums.count(x) for x in set(nums)}
x = max(count, key=count.get)
print(x, count[x])


# Q4
A = {x*7 for x in [1,0,2,4,7]}
B = {x*9 for x in [1,0,2,4,7]}

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
    "name":"Kamlesh",
    "roll_no":"1023833",
    "branch":"CSE",
    "age":19,
    "city":"Mumbai"
}

print(d | friend)
print({k:v for k,v in d.items() if isinstance(v,str)})