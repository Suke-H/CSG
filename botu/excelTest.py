import pandas as pd
import openpyxl

######関数が機能してるかexcelに出力するテスト######
N = 50

#点群から50個もってくる
perm = np.random.permutation(points.shape[0])
data = points[perm[0:N]]

#列にrを追加するための処理
for i in range(N):
    for j in range(9):
        data = np.insert(data, 10*i+1, data[10*i], axis=0)

data = data.T
r_column = np.asarray([(max_norm/11.0)*(i%10+1) for i in range(N*10)])
data = np.insert(data, 3, r_column, axis=0)

print("="*20)
print(data)

#funcをrを追加するための処理
data = data.T
func_column = np.asarray([])

for p in data:
    func_column = np.append(func_column, func(p))

data = data.T
data = np.insert(data, 4, func_column, axis=0)
data = data.T

print(data.shape)


df = pd.DataFrame(data, index=[i+1 for i in range(10*N)], columns=["x0", "y0", "z0", "r", "func"])
df.to_excel('./data/funcTest.xlsx', sheet_name='sample2')
