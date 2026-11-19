
import numpy as np

print("Name: Madhav | Roll No: 1024170356")


# Q1
temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])

print("\nQ1 Original temperatures:", temperature)

# Adding 2 to all temperatures
temp_plus2 = temperature + 2
print("a) After adding 2:", temp_plus2)

# Converting Celsius to Fahrenheit
temp_f = (9/5) * temperature + 32
print("b) Fahrenheit:", temp_f)

# Finding temperatures above 32
above_32 = temperature[temperature > 32]
print("c) Temperatures above 32:", above_32)

# Counting temperatures above 32
count = np.sum(temperature > 32)
print("d) Count:", count)

print("e) Vectorization is faster because it performs operations on the whole array at once instead of using loops. Boolean indexing also makes filtering values easier and faster.")


# Q2
steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])

print("\nQ2 Steps matrix:\n", steps)

print("a) Total steps:", np.sum(steps))
print("b) Average steps:", np.mean(steps))
print("c) Maximum steps:", np.max(steps))
print("   Minimum steps:", np.min(steps))

print("d) Total steps per day:", np.sum(steps, axis=0))
print("e) Total steps per user:", np.sum(steps, axis=1))

# Finding the position of maximum value
index = np.argmax(steps)
row, col = np.unravel_index(index, steps.shape)

print("f) Maximum value:", steps[row, col])
print("   User index:", row, "Day index:", col)


# Q3
original = np.array([1, 2, 3, 4, 5, 6])

print("\nQ3 Original array:", original)

# Taking a slice of the array
subset = original[1:5]
print("b) Subset:", subset)

# Changing subset also changes original array
subset[0] = 999
print("c) Original after changing subset:", original)
print("   Subset:", subset)

# Making a copy of the array
copied = original[1:5].copy()
copied[0] = 500

print("d) Original after changing copy:", original)
print("   Copied array:", copied)

# Creating a 3x4 matrix
matrix = np.arange(1, 13).reshape(3, 4)
print("e) 3x4 Matrix:\n", matrix)

# Accessing different rows and columns
print("f) First row:", matrix[0])
print("   Last row:", matrix[-1])
print("   Second column:", matrix[:, 1])
print("   Selected elements:\n", matrix[1:3, 2:4])

# Using flatten and ravel
flat_copy = matrix.flatten()
flat_view = matrix.ravel()

print("g) Flatten:", flat_copy)
print("   Ravel:", flat_view)

# Changing ravel affects original matrix
flat_view[0] = 100
print("h) Matrix after changing ravel:\n", matrix)

# Changing flatten does not affect original matrix
flat_copy[1] = 200
print("i) Matrix after changing flatten:\n", matrix)

# Displaying array properties
print("j) Shape:", matrix.shape)
print("   Dimensions:", matrix.ndim)
print("   Size:", matrix.size)
print("   Data type:", matrix.dtype)


# Q4
X = np.array([
    [6, 70, 3],
    [5, 50, 6],
    [8, 80, 2],
    [4, 30, 8]
])

y = np.array([40, 65, 30, 85])

# Checking shape and dimensions
print("\nQ4")
print("a) Shape of X:", X.shape)
print("   Dimensions:", X.ndim)

# Transpose of X
print("b) Transpose of X:\n", X.T)

# Multiplying transpose with original matrix
XtX = X.T @ X
print("c) X.T @ X:\n", XtX)

# Finding inverse
XtX_inv = np.linalg.inv(XtX)
print("d) Inverse:\n", XtX_inv)

# Calculating regression coefficients
Xty = X.T @ y
beta = XtX_inv @ Xty

print("e) X.T @ y:", Xty)
print("   Beta values:", beta)

print("Interpretation:")
print("Beta[0] shows the effect of sleep on assistance score.")
print("Beta[1] shows the effect of activity level on assistance score.")
print("Beta[2] shows the effect of stress level on assistance score.")

# Predicting score for a new user
new_user = np.array([5, 40, 7])
predicted = new_user @ beta

print("f) New user:", new_user)
print("   Predicted assistance score:", predicted)
