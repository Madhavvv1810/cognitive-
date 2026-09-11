# Q1
print("Madhav")
print("Madhav")
print("1024170356")


# Q2.1
a = 25
b = 43
c = 92
print(a+b+c)

# Q2.2
a = "Good"
b = " Morning"
c = " Everyone"
print(a+b+c)


# Q3.1
a = input("Enter first string: ")
b = input("Enter second string: ")
print(a+b)

# Q3.2
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print(a+b)


# Q4.1
for i in range(1,11):
    print(7*i)

for i in range(1,11):
    print(9*i)

# Q4.2
n = int(input("Enter number: "))
for i in range(1,11):
    print(n*i)

# Q4.3
n = int(input("Enter number: "))
print(sum(range(1,n+1)))


# Q5.1
a = int(input())
b = int(input())
c = int(input())
print(max(a,b,c))

# Q5.2
n = int(input())
s = 0
for i in range(1,n+1):
    if i%7 == 0 and i%9 == 0:
        s += i
print(s)

# Q5.3
n = int(input())
s = 0
for i in range(2, n+1):
    prime = True

    for j in range(2, i):
        if i % j == 0:
            prime = False
            break
    if prime:
        s += i
print(s)

# Q6.1
def AddOddNumbers(n):
    return sum(range(1,n+1,2))

n = int(input())
print(AddOddNumbers(n))

# Q6.2
def AddPrimeNumbers(n):
    s = 0
    for i in range(2,n+1):
        if all(i%j != 0 for j in range(2,i)):
            s += i
    return s

n = int(input())
print(AddPrimeNumbers(n))


# Q7
import math

print(math.exp(-200))
print(math.log(100,2))
print(math.log(100,10))
print(math.log10(100))
print(math.cos(30))
print(math.sin(30))
print(math.tan(30))
print(math.sqrt(324))
print(math.ceil(89.9))
print(math.floor(89.9))


# Q8.1
s = "Hello World!"
print(s)
print(s[0])
print(s[1:5])
print(s[:-5])

# Q8.2
s = "Hello World!"
print(len(s))
print(s.upper())
print(s.lower())

# Q8.3
name = input()
age = int(input())
price = float(input())
print(name, age, price)

# Q8.4
s = """This is a long string
with multiple lines."""
print(s)

# Q8.5
s = " Indian   Army    "
print(len(s))
print(s.strip())
print(len(s.strip()))

# Q8.6
s = " Indian,   Army    "
print(s.split())
print(s.split(","))
print(s.strip().split(","))

# Q8.7
s = " Indian Army    "
print(s.count(" "))
print(s.count("a"))
print(s.count("an"))

# Q8.8
s = "Indian Army"
print(s[::1])
print(s[::2])
print(s[::-1])
print(s[::-2])

# Q8.9
for s in ["Indian Army","malayalam","madam","teacher"]:
    print(s == s[::-1])
