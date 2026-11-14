"""
诊断：为什么"1/6 域场复制 6 份"看起来对称，代码实现却不对称？
================================================================
思路：若算法概念正确，则位移场的**每一个组成量**都应在对称群下协变：
  标量 dev / m / min_bary : 对称点取值应相等
  矢量 C(参照面心) / u_rad / u_edge / u_dir : 应满足 u[g(p)] = g·u[p]
逐个检验，不协变的那个量 = 实现 bug 所在。

群：
  (a) face-0 稳定子群 D3 = 3 个绕面心旋转 + 3 个过顶点中线镜像
      —— 正是用户把 1/6 三角形复制 6 份所用的操作
  (b) 完整 120 阶二十面体群 = 60 proper + 60 improper(-g)

再用 Reynolds 群平均把方向场投影到对称子空间，验证对称性是否可恢复，并测 rho_max。
"""
import numpy as np, math, json, os
from scipy.spatial import cKDTree
import improve_corrected as ic

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>0 else v

def rot_axis(k,theta):
    k=normalize(k); c=math.cos(theta); s=math.sin(theta)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

# ---------------- 数据：完全复用正式实现 ----------------
n=27
P0,fi,I,J,isedge = ic.gen_meta(n)
C,dev,is_vertex,u_edge,min_bary,Cs = ic.build_fields(P0,fi)
corr = ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
cache="E:/毕业论文/算法/corrected_coef_n27.json"
raw = json.load(open(cache))["raw"] if os.path.exists(cache) else [0.033,0.064,0.0002]
m = corr.m_of(raw)
u_rad = corr.u_rad
u_dir = np.array([corr.dir_of(k) for k in range(len(P0))])
N=len(P0); tree=cKDTree(P0)
mask = ~is_vertex                      # 顶点固定不动，方向无意义，排除
print(f"N={N}  顶点数={is_vertex.sum()}  raw={np.round(raw,6)}")

# ---------------- 群 ----------------
V=ic.V
axis_v=normalize(V[1]); axis_f=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(axis_v,2*math.pi/5),rot_axis(axis_v,-2*math.pi/5),
      rot_axis(axis_f,2*math.pi/3),rot_axis(axis_f,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); k=tuple(np.round(M.ravel(),8))
    if k in seen: continue
    seen.add(k); G60.append(M)
    for g in gens: todo.append(g@M)
G120 = G60 + [-g for g in G60]
print(f"proper rotations = {len(G60)},  full group = {len(G120)}")

# 校验群元确实保 12 顶点集
tv=cKDTree(V)
bad=sum(1 for g in G120 if tv.query((g@V.T).T)[0].max()>1e-6)
print(f"不保顶点集的群元数 = {bad}")

# face 0 稳定子群 D3（3 旋转 + 3 镜像）—— 用户拼 6 份
f0=ic.faces[0]; A,B,Cc=V[f0[0]],V[f0[1]],V[f0[2]]
C0=normalize(A+B+Cc)
D3=[]
for k in range(3): D3.append((f"R3_{120*k}deg", rot_axis(C0,2*math.pi*k/3)))
for name,vert in zip(["A","B","C"],[A,B,Cc]):
    nn=normalize(np.cross(C0,vert)); S=np.eye(3)-2*np.outer(nn,nn)
    D3.append((f"Mirror_{name}",S))

def idx_of(g):
    d,j=tree.query((g@P0.T).T)   # query 返回 (距离, 索引)
    return j,float(d.max())

def diagnose(label,ops):
    print(f"\n===== {label} =====")
    cache_idx=[(nm,g,*idx_of(g)) for nm,g in ops]
    print(f"  点集自映射最大误差 = {max(x[3] for x in cache_idx):.3e}  (应 ~0)")
    for sname,s in [("dev",dev),("m",m),("min_bary",min_bary)]:
        w=max(float(np.abs(s[j][mask]-s[mask]).max()) for _,_,j,_ in cache_idx)
        print(f"  标量 {sname:>10s} 最大偏差 = {w:.8f}")
    for vname,u in [("C(参照面心)",C),("u_rad",u_rad),("u_edge",u_edge),("u_dir(最终)",u_dir)]:
        worst=-1; wname=""; wk=None
        for nm,g,j,_ in cache_idx:
            gu=(g@u.T).T
            num=np.sum(gu[mask]*u[j][mask],axis=1)
            den=np.linalg.norm(gu[mask],axis=1)*np.linalg.norm(u[j][mask],axis=1)+1e-30
            cos=np.clip(num/den,-1,1)
            dd=np.degrees(np.arccos(cos))
            i=int(np.argmax(dd)); d=float(dd[i])
            if d>worst: worst,wname,wk=d,nm,(np.flatnonzero(mask)[i], j[np.flatnonzero(mask)[i]])
        print(f"  矢量 {vname:>14s} 最大角偏差 = {worst:8.3f}deg  (最坏操作 {wname})")
        if vname.startswith("u_dir") and wk:
            k,q=wk
            print(f"      最坏点 k={k} q={q}: fi={fi[k]}/{fi[q]}  I,J={I[k]},{J[k]}  isedge={isedge[k]}"
                  f"  min_bary={min_bary[k]:.4f}  dev={dev[k]:.3f}deg")
    return cache_idx

d3_idx = diagnose("face-0 稳定子群 D3 (= 用户拼 6 份的操作)", D3)
all_idx= diagnose("完整 120 阶二十面体群", [(f"g{i}",g) for i,g in enumerate(G120)])

# ---------------- Reynolds 群平均：把方向场投影到对称子空间 ----------------
print("\n===== Reynolds 群平均（对称化方向场） =====")
acc=np.zeros((N,3))
for g in G120:
    j,_=idx_of(g)
    acc += (g.T @ u_dir[j].T).T          # g^{-1} · u[g(p)]
mag=np.linalg.norm(acc,axis=1)
u_sym=np.array([normalize(a) for a in acc])
print(f"  |acc| 最小={mag.min():.5f}  <1e-6 的点数={(mag<1e-6).sum()}  (轴上的点被抵消=>零位移)")
cos=np.clip(np.sum(u_sym[mask]*u_dir[mask],axis=1),-1,1)
print(f"  原始场 vs 对称化场: 平均角差={np.degrees(np.arccos(cos)).mean():.3f}deg  "
      f"最大={np.degrees(np.arccos(cos.min())):.3f}deg")

def build_P(u,scale=1.0):
    P=np.empty_like(P0)
    for k in range(N):
        if is_vertex[k] or abs(m[k])<1e-9 or np.linalg.norm(u[k])<1e-9:
            P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k], scale*m[k], normalize(u[k]))
    return P

def sym_dev(P):
    t=cKDTree(P); w=0.0
    for g in G120:
        w=max(w,float(t.query((g@P.T).T)[0].max()))
    return math.degrees(w)

print(f"  sym_dev(P0 原始点集)      = {sym_dev(P0):.9f}deg")
P_orig=build_P(u_dir); print(f"  sym_dev(原代码场)          = {sym_dev(P_orig):.9f}deg")
P_sym=build_P(u_sym)
print(f"  sym_dev(对称化场)          = {sym_dev(P_sym):.9f}deg   <-- 应 ~0")
print(f"  rho_max(P0)                = {ic.rho(P0):.6f}deg")
print(f"  rho_max(原代码场)          = {ic.rho(P_orig):.6f}deg")
print(f"  rho_max(对称化场, 原 m)    = {ic.rho(P_sym):.6f}deg")

print("\n  m 缩放扫描（对称化场）:")
for s in [0.8,0.9,1.0,1.1,1.2,1.3,1.5]:
    P=build_P(u_sym,s)
    print(f"    scale={s:.2f}  rho_max={ic.rho(P):.6f}deg  <=1.5? {'YES' if ic.rho(P)<=1.5 else 'NO'}")

json.dump({"D3":{nm:float(idx_of(g)[1]) for nm,g,_,_ in d3_idx}},
          open("E:/毕业论文/算法/diag_covariance.json","w"),indent=2)
print("\n[完成] diag_covariance.json")
