"""
决定性验证：
  Part 1 确认 C_cov 不协变是否 ONLY 发生在"到多个面心等距(tie)"的对称轴点
  Part 2 用【矢量 Reynolds 群平均】构造严格 120 阶协变的位移场
         (= 用户"1/6 域定义 + 对称复制 6 份"思路的全球严格版)
         测 sym_dev 与 rho_max，回答"修好对称后覆盖还达不达标"
  Part 3 与"仅 D3(单面 6 折)复制"对照
"""
import numpy as np, math, json
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
P0,fi,I,J,isedge=ic.gen_meta(n)
C,dev,is_vertex,u_edge,min_bary,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
raw=json.load(open("E:/毕业论文/算法/corrected_coef_n27.json"))["raw"]
m=corr.m_of(raw)
u_dir=np.array([corr.dir_of(k) for k in range(len(P0))])
N=len(P0); tree=cKDTree(P0)

# ---- 群 ----
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
f0=faces[0]; A,B,Cc=V[f0[0]],V[f0[1]],V[f0[2]]; C0=normalize(A+B+Cc)
D3=[rot_axis(C0,2*math.pi*k/3) for k in range(3)]
for vert in [A,B,Cc]:
    nn=normalize(np.cross(C0,vert)); D3.append(np.eye(3)-2*np.outer(nn,nn))

def idx_of(g):
    d,j=tree.query((g@P0.T).T); return j
IDX120=[idx_of(g) for g in G120]
IDX_D3=[idx_of(g) for g in D3]

def cov_angle(u,IDX,G):
    msk=~is_vertex; w=0.0; worst_k=None; worst_g=None
    for gi,(g,j) in enumerate(zip(G,IDX)):
        gu=(g@u.T).T
        num=np.sum(gu[msk]*u[j][msk],axis=1)
        den=np.linalg.norm(gu[msk],axis=1)*np.linalg.norm(u[j][msk],axis=1)+1e-30
        dd=np.degrees(np.arccos(np.clip(num/den,-1,1)))
        if dd.max()>w: w=float(dd.max()); worst_k=int(np.flatnonzero(msk)[dd.argmax()]); worst_g=gi
    return w,worst_k,worst_g

def sym_dev(P):
    t=cKDTree(P); w=0.0
    for g in G120: w=max(w,float(t.query((g@P.T).T)[0].max()))
    return math.degrees(w)

# ================= Part 1: tie 诊断 =================
Cs_arr=np.array([normalize(V[f[0]]+V[f[1]]+V[f[2]]) for f in faces])
D=P0@Cs_arr.T                       # 点积，越大越近
srt=np.sort(D,axis=1)
gap=srt[:,-1]-srt[:,-2]            # 最近与次近面心的间隔；~0 => tie
C_cov=Cs_arr[np.argmax(D,axis=1)]
w,wk,wg=cov_angle(C_cov,IDX120,G120)
print("=== Part 1: tie 诊断 ===")
print(f"  面心指派 tie 点数 (gap<1e-9) = {(gap<1e-9).sum()} / {N}")
print(f"  C_cov 最大协变偏差 = {w:.4f}deg, 最坏点 k={wk}")
print(f"  该点 gap={gap[wk]:.3e}  ->  {'是 tie 点' if gap[wk]<1e-9 else '非 tie 点(!)'}")
tie=gap<1e-9
# 只在非 tie 点上检查协变性
msk=(~is_vertex)&(~tie)
w2=0.0
for g,j in zip(G120,IDX120):
    gu=(g@C_cov.T).T
    num=np.sum(gu[msk]*C_cov[j][msk],axis=1)
    den=np.linalg.norm(gu[msk],axis=1)*np.linalg.norm(C_cov[j][msk],axis=1)+1e-30
    w2=max(w2,float(np.degrees(np.arccos(np.clip(num/den,-1,1))).max()))
print(f"  排除 tie 点后 C_cov 最大偏差 = {w2:.3e}deg  (应 ~0)")

# ================= Part 2: 矢量 Reynolds 群平均 =================
print("\n=== Part 2: 矢量 Reynolds 群平均 (严格协变) ===")
d0=m[:,None]*u_dir                  # 原代码的位移矢量场(单位:度)
d0[is_vertex]=0.0                   # 顶点固定
def reynolds(G,IDX):
    acc=np.zeros((N,3))
    for g,j in zip(G,IDX):
        acc += (g.T @ d0[j].T).T    # g^{-1}·d[g(p)]
    return acc/len(G)
v120=reynolds(G120,IDX120)
vD3 =reynolds(D3,IDX_D3)
def build_P(v,scale=1.0):
    P=np.empty_like(P0); mag=np.linalg.norm(v,axis=1)
    for k in range(N):
        if is_vertex[k] or mag[k]<1e-12: P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],scale*mag[k],normalize(v[k]))
    return P
print(f"  |v120| 范围 = [{np.linalg.norm(v120,axis=1).min():.6f}, {np.linalg.norm(v120,axis=1).max():.6f}] deg")
print(f"  v120 协变偏差 = {cov_angle(v120,IDX120,G120)[0]:.3e}deg   (应 ~0)")
for name,v in [("G120 平均",v120),("D3 平均",vD3)]:
    P=build_P(v)
    print(f"  [{name}] sym_dev={sym_dev(P):.9f}deg   rho_max={ic.rho(P):.6f}deg")
print("  v120 缩放扫描:")
scan=[]
for s in [0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0]:
    P=build_P(v120,s); rr=ic.rho(P); scan.append((s,rr))
    print(f"    scale={s:.2f}  rho_max={rr:.6f}deg  {'<=1.5 OK' if rr<=1.5 else ''}")

# ================= Part 3: 只保留 1/6 基本域的方向再群复制 =================
print("\n=== Part 3: 1/6 基本域定义 + 120 群元复制 (= 用户拼图的全球版) ===")
# 基本域: face 0 内, 方位角 az in [0,60) (以面心 C0 为极点)
ref=normalize(A-np.dot(A,C0)*C0); perp=np.cross(C0,ref)
sel0=(fi==0)
az=np.zeros(N)
for k in np.where(sel0)[0]:
    u=normalize(P0[k]-np.dot(P0[k],C0)*C0)
    a=math.degrees(math.atan2(np.dot(u,perp),np.dot(u,ref)))
    if a<0: a+=360
    az[k]=a
base=(sel0)&(az>=0)&(az<60)
print(f"  1/6 基本域点数 = {base.sum()}  (轨道数约 {N/120:.1f})")
# 用"基本域代表点"的位移，按群复制：v[k] = g · v[rep], 取所有可达代表的平均(等价 Reynolds)
d_base=d0.copy(); d_base[~base]=0.0
acc=np.zeros((N,3)); cnt=np.zeros(N)
for g,j in zip(G120,IDX120):
    acc += (g.T @ d_base[j].T).T
    cnt += (np.linalg.norm(d_base[j],axis=1)>1e-12).astype(float)
v_base=np.zeros((N,3)); ok=cnt>0
v_base[ok]=acc[ok]/cnt[ok,None]     # 按每个点的轨道命中次数归一化
print(f"  轨道命中次数: min={cnt.min():.0f} max={cnt.max():.0f} 未命中={(cnt==0).sum()}")
P=build_P(v_base)
print(f"  基本域复制: sym_dev={sym_dev(P):.9f}deg   rho_max={ic.rho(P):.6f}deg")
for s in [0.5,0.8,1.0,1.2,1.5,2.0]:
    P=build_P(v_base,s)
    print(f"    scale={s:.2f}  rho_max={ic.rho(P):.6f}deg  {'<=1.5 OK' if ic.rho(P)<=1.5 else ''}")

np.save("E:/毕业论文/算法/v120_symmetric.npy",v120)
np.save("E:/毕业论文/算法/v_base_symmetric.npy",v_base)
json.dump({"tie_points":int((gap<1e-9).sum()),"C_cov_deg":w,"C_cov_nontie_deg":w2,
           "v120_cov_deg":cov_angle(v120,IDX120,G120)[0],
           "scan":scan},open("E:/毕业论文/算法/verify_orbit_result.json","w"),indent=2)
print("\n[完成] verify_orbit_result.json")
