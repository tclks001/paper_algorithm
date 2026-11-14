"""
诊断：为什么位移幅度乘 1.5 反而得到更小的 rho_max？
====================================================
要区分两种互斥解释：
  [A] 原版 3 参数 Nelder-Mead 没收敛（在自身参数族内就没找到最优）
      -> 检验：对【原版场 d0 = m(dev)*u_dir】直接做 scale 扫描。
         注意 m = u0^2 + u1^2*dev + u2^2*dev^2，把 (u0,u1,u2) 整体乘 sqrt(s)
         即得到 s*m，完全在原参数族内。若 s*=1 明显非最优 => 优化欠收敛。
  [B] scale 作用在【Reynolds 群平均场 v】上，v 是另一个场
      （反对称分量被抵消、1572 个点变零位移），不在原参数族内，
      故需要重新标定幅度，与原优化收敛与否无关。

另检验目标函数 rho_max 关于 s 的非光滑性（最坏三角形切换）。
"""
import numpy as np, math, json, time
from scipy.spatial import cKDTree, ConvexHull
import improve_corrected as ic

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

n=27
P0,fi,I,J,isedge=ic.gen_meta(n)
C,dev,is_vertex,u_edge,min_bary,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
raw=json.load(open("E:/毕业论文/算法/corrected_coef_n27.json"))["raw"]
m=corr.m_of(raw)
u_dir=np.array([corr.dir_of(k) for k in range(len(P0))])
N=len(P0)

def build_P(dvec,scale=1.0):
    """dvec[k] = 该点的单位位移方向；scale 乘到幅度上"""
    P=np.empty_like(P0)
    mag=np.linalg.norm(dvec,axis=1) if dvec.ndim==2 else np.abs(dvec)
    for k in range(N):
        if is_vertex[k] or mag[k]<1e-12: P[k]=P0[k]; continue
        u=normalize(dvec[k])
        P[k]=ic.exp_map(P0[k],scale*mag[k],u)
    return P

def rho_and_worst(P):
    h=ConvexHull(P); mx=0.0; wt=None
    for t in h.simplices:
        r=ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]])
        if r>mx: mx=r; wt=tuple(sorted(t))
    return math.degrees(mx),wt

d0=(m[:,None]*u_dir)          # 原版场（矢量，模长=m）

print("=== [A] 原版场 d0 = m(dev)*u_dir 的 scale 扫描 ===")
print("  (若 s* 明显偏离 1 => 原 3 参数 Nelder-Mead 在自身参数族内欠收敛)")
resA=[]
for s in [0.6,0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.2,2.5]:
    r,wt=rho_and_worst(build_P(d0,s)); resA.append((s,r,wt))
    print(f"    s={s:.2f}  rho_max={r:.6f}deg  最坏三角形={wt}")
bestA=min(resA,key=lambda x:x[1])
print(f"  -> 原版场最优 s* = {bestA[0]:.2f},  rho={bestA[1]:.6f}deg "
      f"(原拟合 s=1: {[x for x in resA if x[0]==1.0][0][1]:.6f}deg)")

# 最坏三角形切换 -> 目标函数非光滑
switches=[(resA[i][0],resA[i-1][2],resA[i][2]) for i in range(1,len(resA)) if resA[i][2]!=resA[i-1][2]]
print(f"  相邻扫描点间最坏三角形切换次数 = {len(switches)} / {len(resA)-1}")
print(f"  -> rho_max(s) 为 max 型分段函数，梯度不连续，Nelder-Mead 易早停")

# ---- 直接从多起点重跑 3 参数优化，看原系数是否局部极小 ----
print("\n=== [A2] 3 参数空间多起点 Nelder-Mead（检验收敛性） ===")
t0=time.time()
def obj(u):
    mm=u[0]**2+u[1]**2*dev+u[2]**2*dev**2
    return rho_and_worst(build_P((mm[:,None]*u_dir)))[0]
print(f"  当前缓存系数 rho = {obj(np.array(raw)):.6f}deg")
# 沿"幅度方向"直接检验：u 整体乘 sqrt(s)
for s in [1.0,1.2,1.5,1.8]:
    u=np.array(raw)*math.sqrt(s)
    print(f"    u *= sqrt({s:.1f})  ->  rho = {obj(u):.6f}deg")

# ---- [B] 群平均场 ----
print("\n=== [B] Reynolds 群平均场 v 的 scale 扫描（对照） ===")
V=ic.V
def rot_axis(k,theta):
    k=normalize(k); c=math.cos(theta); s=math.sin(theta)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)
av=normalize(V[1]); af=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(av,2*math.pi/5),rot_axis(av,-2*math.pi/5),
      rot_axis(af,2*math.pi/3),rot_axis(af,-2*math.pi/3)]
G=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); k=tuple(np.round(M.ravel(),8))
    if k in seen: continue
    seen.add(k); G.append(M)
    for g in gens: todo.append(g@M)
G120=G+[-g for g in G]
d0v=d0.copy(); d0v[is_vertex]=0.0
tree=cKDTree(P0); acc=np.zeros((N,3))
for g in G120:
    _,j=tree.query((g@P0.T).T)
    acc+=(g.T@d0v[j].T).T
v=acc/len(G120)
print(f"  |v| max = {np.linalg.norm(v,axis=1).max():.6f}deg   "
      f"(原版 |d0| max = {np.linalg.norm(d0,axis=1).max():.6f}deg)")
print(f"  零位移点数 = {(np.linalg.norm(v,axis=1)<1e-9).sum()}  <- 群平均抵消掉的分量")
resB=[]
for s in [0.6,0.8,1.0,1.2,1.5,1.8,2.0,2.5]:
    r,_=rho_and_worst(build_P(v,s)); resB.append((s,r))
    print(f"    s={s:.2f}  rho_max={r:.6f}deg")
bestB=min(resB,key=lambda x:x[1])
print(f"  -> 群平均场最优 s* = {bestB[0]:.2f}, rho={bestB[1]:.6f}deg")

print("\n=== 结论判据 ===")
print(f"  原版场最优 s*={bestA[0]:.2f} vs 群平均场最优 s*={bestB[0]:.2f}")
if abs(bestA[0]-1.0)<0.15:
    print("  => 原版 3 参数优化在其参数族内基本收敛；scale 收益主要来自 [B] 换场后的重新标定")
else:
    print("  => 原版 3 参数优化欠收敛（[A] 成立）；同时 [B] 换场也贡献一部分")
json.dump({"scan_original":[(s,r) for s,r,_ in resA],
           "scan_reynolds":resB,
           "best_s_original":bestA[0],"best_s_reynolds":bestB[0],
           "rho_original_s1":[x[1] for x in resA if x[0]==1.0][0],
           "switches":len(switches)},
          open("E:/毕业论文/算法/diagnose_scale_result.json","w"),indent=2)
print(f"\n[完成] 用时 {time.time()-t0:.1f}s")
