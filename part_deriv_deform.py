import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

filename = "S600_die_attach_deformation.txt"
df = pd.read_csv(filename, sep="\t", lineterminator="\r")
cols = df.columns
print(cols)
print(df.head())
# print(df)

C_NODE = "Node Number"
C_X = "X Location (mm)"
C_Y = "Y Location (mm)"
C_Z = "Z Location (mm)"
C_DEF = "Directional Deformation (mm)"

# print(df.iloc[3][C_X])

# Find bounding box of data
x_min = df.iloc[0][C_X]
y_min = df.iloc[0][C_Y]
x_max = df.iloc[0][C_X]
y_max = df.iloc[0][C_Y]
for index, row in df.iterrows():
    # print(row[C_X], row[C_Y])
    x = float(row[C_X])
    y = float(row[C_Y])
    if y > y_max:
        y_max = y
    if x > x_max:
        x_max = x
    if y < y_min:
        y_min = y
    if x < x_min:
        x_min = x

print(f"Min: ({x_min},{y_min})  Max: ({x_max},{y_max})")

# get measurement axis
x = (x_min + x_max) / 2
y = (y_min + y_max) / 2

# collect deformation along this axis by finding closest node
data = []
p_x = x
for p_y in np.arange(y_min, y_max, 0.1):
    df_closest = (
        df.copy(deep=True)
        .assign(d=(p_x - df[C_X]) ** 2 + (p_y - df[C_Y]) ** 2)
        .sort_values("d")
        .drop("d", axis=1)
    )
    data.append({"y": p_y, "deformation": df_closest.iloc[0][C_DEF]})

df_y = pd.DataFrame(data)

print(df_y)

# get x and y vectors
x = df_y["y"]
y = df_y["deformation"]

# calculate polynomial
z = np.polyfit(x, y, 3)
f = np.poly1d(z)
print(f, f.deriv(m=2))

# calculate new x's and y's
x_new = np.linspace(y_min, y_max, 50)
y_new = f(x_new)

# plt.figure()
df_y.plot(x="y", y="deformation")
plt.plot(x_new, y_new)
plt.plot(x_new, f.deriv(m=1)(x_new))
plt.plot(x_new, f.deriv(m=2)(x_new))
plt.show()
