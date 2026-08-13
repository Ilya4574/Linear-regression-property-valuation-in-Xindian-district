import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_excel("data/Real estate valuation data set.xlsx", index_col=0)

fig_bloxplot_y = px.box(df, y="Y house price of unit area")
fig_bloxplot_y.update_layout(width=800, height=600, font=dict(size=14))
#fig_bloxplot_y.show()

lower = df["Y house price of unit area"].quantile(0.01)
upper = df["Y house price of unit area"].quantile(0.99)

df = df[(df["Y house price of unit area"] >= lower) & (df["Y house price of unit area"] <= upper)]
df = df[df["X3 distance to the nearest MRT station"] <= 6000]

fig_bloxplot_y = px.box(df, y="Y house price of unit area")
fig_bloxplot_y.update_layout(width=800, height=600, font=dict(size=14))
#fig_bloxplot_y.show()

fig_scatter_x1 = px.scatter(
    x=df["X1 transaction date"],
    y=df["Y house price of unit area"],
    labels={"x": "Дата сделки (доли года)", "y": "Цена за единицу площади"},
    title="Рисунок 3 - Зависимость цены от даты сделки",
    opacity=0.6
)
fig_scatter_x1.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x1.show()

fig_scatter_x2 = px.scatter(
    x=df["X2 house age"],
    y=df["Y house price of unit area"],
    labels={"x": "Возраст дома, лет", "y": "Цена за единицу площади"},
    title="Рисунок 4 - Зависимость цены от возраста дома",
    opacity=0.6
)
fig_scatter_x2.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x2.show()

fig_scatter_x3 = px.scatter(
    x=df["X3 distance to the nearest MRT station"],
    y=df["Y house price of unit area"],
    labels={"x": "Расстояние до метро, м", "y": "Цена за единицу площади"},
    title="Рисунок 5 - Зависимость цены от расстояния до метро",
    opacity=0.6
)
fig_scatter_x3.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x3.show()

fig_scatter_x4 = px.scatter(
    x=df["X4 number of convenience stores"],
    y=df["Y house price of unit area"],
    labels={"x": "Количество магазинов", "y": "Цена за единицу площади"},
    title="Рисунок 6 - Зависимость цены от количества магазинов",
    opacity=0.6
)
fig_scatter_x4.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x4.show()

fig_scatter_x5 = px.scatter(
    x=df["X5 latitude"],
    y=df["Y house price of unit area"],
    labels={"x": "Широта", "y": "Цена за единицу площади"},
    title="Рисунок 7 - Зависимость цены от широты",
    opacity=0.6
)
fig_scatter_x5.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x5.show()

fig_scatter_x6 = px.scatter(
    x=df["X6 longitude"],
    y=df["Y house price of unit area"],
    labels={"x": "Долгота", "y": "Цена за единицу площади"},
    title="Рисунок 8 - Зависимость цены от долготы",
    opacity=0.6
)
fig_scatter_x6.update_layout(width=800, height=600, font=dict(size=14))
#fig_scatter_x6.show()

# Модель со всеми признаками
x_raw = df[["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores", "X5 latitude", "X6 longitude"]].values
y = df["Y house price of unit area"].values

X = np.hstack([np.ones((x_raw.shape[0], 1)), x_raw])

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

split_idx = int(0.8*len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

coef = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

b0, b1, b2, b3, b4, b5, b6 = coef

y_pred = X_test @ coef

mae = sum(abs(y_test - y_pred)) / len(y_test)

rmse = (sum((y_test - y_pred)**2) / len(y_test)) ** 0.5
ss_res = sum((y_test - y_pred)**2)
ss_tot = sum((y_test - np.mean(y_train))**2)
r2 = 1 - ss_res/ss_tot


# Модель с X3, X4
x_raw = df[["X3 distance to the nearest MRT station", "X4 number of convenience stores"]].values
y = df["Y house price of unit area"].values

X = np.hstack([np.ones((x_raw.shape[0], 1)), x_raw])

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

split_idx = int(0.8*len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

coef = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

b0, b1, b2 = coef

y_pred = X_test @ coef

mae = sum(abs(y_test - y_pred)) / len(y_test)

rmse = (sum((y_test - y_pred)**2) / len(y_test)) ** 0.5

ss_res = sum((y_test - y_pred)**2)
ss_tot = sum((y_test - np.mean(y_train))**2)
r2 = 1 - ss_res/ss_tot

print(mae, rmse, r2)


# Модель с X2, X3, X4
x_raw = df[["X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores"]].values
y = df["Y house price of unit area"].values

X = np.hstack([np.ones((x_raw.shape[0], 1)), x_raw])

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

split_idx = int(0.8*len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

coef = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

b0, b1, b2, b3 = coef

y_pred = X_test @ coef

mae = sum(abs(y_test - y_pred)) / len(y_test)

rmse = (sum((y_test - y_pred)**2) / len(y_test)) ** 0.5

ss_res = sum((y_test - y_pred)**2)
ss_tot = sum((y_test - np.mean(y_train))**2)
r2 = 1 - ss_res/ss_tot
