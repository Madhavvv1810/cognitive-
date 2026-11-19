import numpy as np

print("Name: Madhav | Roll No: 1024170356")

# ------------------------------------------------------------
# Q1
# ------------------------------------------------------------
a = np.array([1, 2, 3, 4, 5])
print("\nQ1 Array:", a)

print("a) Addition of 2:", a + 2)
print("b) Multiply by 3:", a * 3)
print("c) Divide by 2:", a / 2)


# ------------------------------------------------------------
# Q2
# ------------------------------------------------------------
# a) Reverse array
arr = np.array([1, 2, 3, 6, 4, 5])
print("\nQ2a Reversed array:", arr[::-1])

# b) Most frequent value and indices
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
vals, counts = np.unique(x, return_counts=True)
max_count = counts.max()
freq_vals = vals[counts == max_count]

print("\nQ2b(i) Array x:", x)
print("Most frequent value(s):", freq_vals, "Count:", max_count)
for v in freq_vals:
    print("Value", v, "indices:", np.where(x == v)[0])

y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
vals, counts = np.unique(y, return_counts=True)
max_count = counts.max()
freq_vals = vals[counts == max_count]

print("\nQ2b(ii) Array y:", y)
print("Most frequent value(s):", freq_vals, "Count:", max_count)
for v in freq_vals:
    print("Value", v, "indices:", np.where(y == v)[0])


# ------------------------------------------------------------
# Q3
# Correct 2-D array syntax: np.array([[ ... ], [ ... ], [ ... ]])
# ------------------------------------------------------------
arr2 = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("\nQ3 Array:\n", arr2)
print("a) 1st row, 2nd column =", arr2[0, 1])
print("b) 3rd row, 1st column =", arr2[2, 0])


# ------------------------------------------------------------
# Q4
# ------------------------------------------------------------
Madhav = np.linspace(10, 100, 25)

print("\nQ4 Array Madhav:\n", Madhav)
print("Dimensions/ndim:", Madhav.ndim)
print("Shape:", Madhav.shape)
print("Total elements/size:", Madhav.size)
print("Data type:", Madhav.dtype)
print("Total bytes:", Madhav.nbytes)

# Reshape to 5x5 and then transpose
Madhav_2d = Madhav.reshape(5, 5)
print("\nReshaped to 5x5:\n", Madhav_2d)
print("Transpose using T attribute:\n", Madhav_2d.T)

# For a 1-D array, .T does not change it
print("Original 1-D shape:", Madhav.shape)
print("1-D .T shape:", Madhav.T.shape)
print("Is 1-D .T same as original?", np.array_equal(Madhav, Madhav.T))


# ------------------------------------------------------------
# Q5
# ------------------------------------------------------------
ucs420_Madhav = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 15, 20, 35]
])

print("\nQ5 Array ucs420_Madhav:\n", ucs420_Madhav)
print("Mean:", ucs420_Madhav.mean())
print("Median:", np.median(ucs420_Madhav))
print("Max:", ucs420_Madhav.max())
print("Min:", ucs420_Madhav.min())
print("Unique elements:", np.unique(ucs420_Madhav))

reshaped_ucs420_Madhav = ucs420_Madhav.reshape(4, 3)
print("\nreshaped_ucs420_Madhav:\n", reshaped_ucs420_Madhav)

resized_ucs420_Madhav = np.resize(ucs420_Madhav, (2, 3))
print("\nresized_ucs420_Madhav:\n", resized_ucs420_Madhav)
i want to put this on git first i have to create a folder on my laptop or whaT?