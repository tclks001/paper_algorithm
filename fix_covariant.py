"""
最小修复验证：把方向场从"依赖面内顶点顺序的叉积"改回"纯几何、对称协变"的定义
==========================================================================
诊断结论（diag_covariance.py）：dev / m / min_bary 全部协变(偏差 0)，
只有 C / u_rad / u_edge / u_dir 不协变(最大 180°)。故算法概念无误，bug 全在方向实现。

improve_corrected.py 的两处 bug：
  [1] 第 71-73 行  C[k] = Cs[fi[k]]
      fi 由"去重时第一个生成的面"决定，不随对称群协变 -> 参照面心偏心 41.8°
  [2] 第 81-89 行  u_edge[k] = normalize(p × (X×Y))
      (X,Y) 取自该面顶点存储顺序 [(A,B),(B,C),(C,A)]；交换两顶点或用镜像时
      叉积法向翻号，而代码没有补偿 -> 镜像/2重旋转下切向反号(180°)

修复（保持"内部径向、近棱切向、顶点固定"的算法概念不变）：
  C      = 20 个面心中与 p 角距最小者（非棱点唯一；协变）
  u_edge = 沿大圆指向"最近棱中点" F_e=normalize(X+Y)：
           normalize(F_e - (F_e·p)p)   —— 纯几何，不依赖任何顶点顺序 / det
  u_rad  = normalize(C - (C·p)p)
  min_bary 保留（实测协变）
符号 sign=±1 表示"朝棱中点" / "背离棱中点"，两者都测。
"""
import numpy as np, math, json, os
from scipy.spatial import cKDTree
import improve_corrected as ic

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
def rot_axis(k,theta):
    k=normalize(k); c=math.cos(theta); s=math.sin(theta)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

V=ic.V; faces=ic.faces
n=27
P0,fi,I,J,isedge = ic.gen_meta(n)
C_old,dev,is_vertex,u_edge_old,min_bary,Cs = ic.build_fields(P0,fi)
raw=json.load(open("E:/毕业论文/算法/corrected_coef_n27.json"))["raw"]
u0,u1,u2=raw; p0,q0,r0=u0*u0,u1*u1,u2*u2
m = p0 + q0*dev + r0*dev**2
N=len(P0); band=0.08
wb=np.clip((band-min_bary)/band,0.0,1.0)

# ---------- 群 ----------
axis_v=normalize(V[1]); axis_f=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(axis_v,2*math.pi/5),rot_axis(axis_v,-2*math.pi/5),
      rot_axis(axis_f,2*math.pi/3),rot_axis(axis_f,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); k=tuple(np.round(M.ravel(),8))
    if k in seen: continue
    seen.add(k); G60.append(M)
    for g in gens: todo.append(g@M)
G120=G60+[-g for g in G60]

def idx_of(g,tree):
    d,j=tree.query((g@P0.T).T); return j

def cov_angle(u,tree):
    """方向场 u 在 G120 下的最大协变角偏差（排除顶点）"""
    msk=~is_vertex; worst=0.0
    for g in G120:
        j=idx_of(g,tree); gu=(g@u.T).T
        num=np.sum(gu[msk]*u[j][msk],axis=1)
        den=np.linalg.norm(gu[msk],axis=1)*np.linalg.norm(u[j][msk],axis=1)+1e-30
        worst=max(worst,float(np.degrees(np.arccos(np.clip(num/den,-1,1))).max()))
    return worst

def sym_dev(P):
    t=cKDTree(P); w=0.0
    for g in G120: w=max(w,float(t.query((g@P.T).T)[0].max()))
    return math.degrees(w)

# ---------- 修复 1: 协变参照面心 ----------
Cs_arr=np.array([normalize(V[f[0]]+V[f[1]]+V[f[2]]) for f in faces])
C_cov=Cs_arr[np.argmax(P0@Cs_arr.T,axis=1)]          # 点积最大 = 角距最小
u_rad_cov=np.array([normalize(C_cov[k]-np.dot(C_cov[k],P0[k])*P0[k]) for k in range(N)])

# ---------- 修复 2: 协变沿棱切向 = 指向最近棱中点 ----------
ep=set()
for f in faces:
    for a,b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]: ep.add(tuple(sorted((a,b))))
ep=sorted(ep)
EX=np.array([V[a] for a,b in ep]); EY=np.array([V[b] for a,b in ep])
EF=np.array([normalize(V[a]+V[b]) for a,b in ep])
aX=np.arccos(np.clip(P0@EX.T,-1,1)); aY=np.arccos(np.clip(P0@EY.T,-1,1))
aXY=np.arccos(np.clip(np.sum(EX*EY,axis=1),-1,1))
Ei=np.argmin(np.abs(aX+aY-aXY[None,:]),axis=1)
Fk=EF[Ei]
u_edge_cov=np.array([normalize(Fk[k]-np.dot(Fk[k],P0[k])*P0[k]) for k in range(N)])

tree=cKDTree(P0)
print("=== 修复后各量的协变角偏差（应 ~0） ===")
print(f"  C_cov      = {cov_angle(C_cov,tree):.6f}deg")
print(f"  u_rad_cov  = {cov_angle(u_rad_cov,tree):.6f}deg")
print(f"  u_edge_cov = {cov_angle(u_edge_cov,tree):.6f}deg")

def build_dir(sign):
    u=np.array([normalize((1-wb[k])*u_rad_cov[k]+wb[k]*(sign*u_edge_cov[k])) for k in range(N)])
    return u

def build_P(u,scale=1.0):
    P=np.empty_like(P0)
    for k in range(N):
        if is_vertex[k] or abs(m[k])<1e-9 or np.linalg.norm(u[k])<1e-12:
            P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],scale*m[k],normalize(u[k]))
    return P

print(f"\n=== 原代码场（对照） ===")
corr=ic.Corrector(P0,fi,C_old,dev,is_vertex,u_edge_old,isedge,min_bary)
P_old=corr.apply(raw)
print(f"  u_dir 协变偏差 = {cov_angle(np.array([corr.dir_of(k) for k in range(N)]),tree):.3f}deg")
print(f"  sym_dev        = {sym_dev(P_old):.9f}deg")
print(f"  rho_max        = {ic.rho(P_old):.6f}deg")

results={}
for sign in [1,-1]:
    u=build_dir(sign)
    print(f"\n=== 修复场 sign={sign:+d} ({'朝棱中点' if sign>0 else '背离棱中点'}) ===")
    print(f"  u_dir 协变偏差 = {cov_angle(u,tree):.6f}deg")
    P=build_P(u)
    print(f"  sym_dev        = {sym_dev(P):.9f}deg   <-- 应 ~0")
    print(f"  rho_max(scale=1)= {ic.rho(P):.6f}deg")
    scan=[]
    for s in [0.6,0.8,1.0,1.2,1.4,1.6,2.0]:
        Ps=build_P(u,s); rr=ic.rho(Ps); scan.append((s,rr))
        print(f"    scale={s:.2f}  rho_max={rr:.6f}deg  {'<=1.5 OK' if rr<=1.5 else ''}")
    results[str(sign)]={"cov_deg":cov_angle(u,tree),"sym_dev_deg":sym_dev(P),
                        "rho1":ic.rho(P),"scan":scan}
json.dump(results,open("E:/毕业论文/算法/fix_covariant_result.json","w"),indent=2)
print("\n[完成] fix_covariant_result.json")
