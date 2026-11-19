import numpy as np

print("Name: Madhav Keshav Krish | Roll No: 1024170356")

# ============================================================
# Q1. Sensor readings
# ============================================================
temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])
print("\nQ1 Original temperatures:", temperature)

# a) Add 2°C via vectorization
temp_plus2 = temperature + 2
print("a) After adding 2°C:", temp_plus2)

# b) Celsius -> Fahrenheit : F = (9/5)C + 32
temp_f = (9 / 5) * temperature + 32
print("b) In Fahrenheit:", temp_f)

# c) Boolean indexing: readings > 32
above_32 = temperature[temperature > 32]
print("c) Readings > 32°C:", above_32)

# d) Count readings > 32
count_above_32 = np.sum(temperature > 32)
print("d) Count > 32°C:", count_above_32)

print("""e) Efficiency explanation:
   Vectorization performs the operation on the entire array at once in
   optimized C code, so no Python loop overhead is incurred. Boolean
   indexing creates a single boolean mask and filters all values in one
   step. Both are faster, shorter, and less error-prone than iterating
   element-by-element in Python.""")


# ============================================================
# Q2. Daily steps matrix
# ============================================================
steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])
print("\nQ2 Steps matrix:\n", steps)

print("a) Total steps:", steps.sum())
print("b) Mean steps:", steps.mean())
print("c) Max:", steps.max(), "| Min:", steps.min())
print("d) Total per day (axis=0):", steps.sum(axis=0))
print("e) Total per user (axis=1):", steps.sum(axis=1))

# f) Position of maximum using argmax + unravel_index
flat_idx = np.argmax(steps)
row, col = np.unravel_index(flat_idx, steps.shape)
print("f) Max value", steps[row, col],
      "at user index", row, ", day index", col)


# ============================================================
# Q3. Slicing, views, copies, reshaping
# ============================================================
original = np.array([1, 2, 3, 4, 5, 6])
print("\nQ3 Original:", original)

# b) Slice (view)
subset = original[1:5]
print("b) subset = original[1:5]:", subset)

# c) Modify subset -> original changes (view)
subset[0] = 999
print("c) After subset[0]=999 -> original:", original)
print("                        subset  :", subset)

# d) Copy slice -> original unchanged
copied = original[1:5].copy()
copied[0] = 500
print("d) After copied[0]=500 -> original:", original)
print("                        copied  :", copied)

# e) 1..12 reshaped to 3x4
matrix = np.arange(1, 13).reshape(3, 4)
print("\ne) 3x4 matrix:\n", matrix)

# f) Indexing / slicing
print("f) First row      :", matrix[0])
print("   Last row       :", matrix[-1])
print("   Second column  :", matrix[:, 1])
print("   Rows 1-2, Cols 2-3:\n", matrix[1:3, 2:4])

# g) flatten() and ravel()
flat_copy = matrix.flatten()   # always a copy
flat_view = matrix.ravel()     # a view when possible
print("g) flatten():", flat_copy)
print("   ravel()  :", flat_view)

# h) Modify ravel() result -> original changes (view)
flat_view[0] = 100
print("h) After ravel()[0]=100, original matrix:\n", matrix)

# i) Modify flatten() result -> original unchanged (copy)
flat_copy[1] = 200
print("i) After flatten()[1]=200, original matrix still:\n", matrix)

# j) Properties
print("j) shape:", matrix.shape, "| ndim:", matrix.ndim,
      "| size:", matrix.size, "| dtype:", matrix.dtype)


# ============================================================
# Q4. OLS regression for cognitive-assistive system
# ============================================================
X = np.array([
    [6, 70, 3],
    [5, 50, 6],
    [8, 80, 2],
    [4, 30, 8]
])
y = np.array([40, 65, 30, 85])

# a) Shape / dimensions
print("\nQ4 a) X.shape:", X.shape, "| X.ndim:", X.ndim)

# b) Transpose
print("b) X.T:\n", X.T)
print("   X.T shape:", X.T.shape,
      "-> rows become features, columns become users.")

# c) X.T @ X
XtX = X.T @ X
print("c) X.T @ X:\n", XtX)

# d) Inverse
XtX_inv = np.linalg.inv(XtX)
print("d) inv(X.T @ X):\n", XtX_inv)

# e) OLS coefficients: beta = (X.T X)^-1 X.T y
Xty = X.T @ y
beta = XtX_inv @ Xty
print("e) X.T @ y:", Xty)
print("   beta (OLS coefficients):", beta)
print("""   Interpretation:
     beta[0] ~ change in predicted assistance score per extra hour
               of sleep (holding others constant)
     beta[1] ~ change in score per extra unit of activity level
     beta[2] ~ change in score per extra unit of stress level""")

# f) Prediction for a new user
new_user = np.array([5, 40, 7])
predicted = new_user @ beta
print("f) New user:", new_user, "-> predicted assistance score:", predicted)