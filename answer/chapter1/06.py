x = "paraparaparadise"
y = "paragraph"

# bi-gram -> n=2
n = 2
x_set = set([x[i : i + 2] for i in range(len(x) - n + 1)])
y_set = set([y[i : i + 2] for i in range(len(y) - n + 1)])

# 和集合
print(f"Union of X and Y : {x_set | y_set}")

# 積集合
print(f"Intersection of X and Y : {x_set & y_set}")

# 差集合
print(f"Difference of X and Y : {x_set - y_set}")

print("Yes" if "se" in x_set else "No")
print("Yes" if "se" in y_set else "No")
