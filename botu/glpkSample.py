import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import numpy as np

#係数行列の次元
N=2
M=2

#係数ベクトル，係数行列の作成
b = np.array([2,3], dtype="float")
A = np.array([[4,3],[1,2]], dtype="float")
c = np.array([120,60], dtype="float")

#決定変数の設定
x = np.zeros(shape=(N), dtype="float")

#決定変数初期化用関数---------------------
xx = {i:x[i] for i in range(len(x))}
def init_x(model,i):
    return xx[i]
# -----------------------------------

#index list
N_index_list = list(range(N))
M_index_list = list(range(M))

#不等式制約
def const(model, j):
    tmp = sum(A[j,i]*model.x[i] for i in N_index_list)
    return tmp <= c[j]

#目的関数
def func(model):
    return sum(b[i] * model.x[i] for i in N_index_list)  

#モデル
model = pyo.AbstractModel("LP-sample")

#決定変数をモデルに組み込み
model.x = pyo.Var(N_index_list, domain=pyo.NonNegativeReals, initialize=init_x)

#目的関数をモデルに組み込む
model.value = pyo.Objective(rule=func, sense=pyo.maximize)

#制約をモデルに組み込み
model.const = pyo.Constraint(M_index_list, rule=const)

#ソルバーの選択とオプション
solver_name = "glpk"
opt = SolverFactory(solver_name)

instance =  model.create_instance()
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #双対変数取得用に必要
results = opt.solve(instance, tee=True)#tee=Trueでログ出力
instance.load(results)
results.write()

#解の概要を見る
instance.display()