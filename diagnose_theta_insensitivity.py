"""
诊断：为什么偏转函数 W 对 rho_max 不敏感？
==========================================
主实验 optimize_theta_joint.py 的结果：6/7 个函数族收敛到完全相同的
rho=1.496312 与 a=0.0041742，尽管 W 形状差别极大。本脚本回答三件事：

  A) 量化 Arad/Atan 相对大小（Reynolds 是否把切向场压没了）
  B) 固定 a，扫描极端 W（w=0 纯径向 / w=1 纯切向 / 各种衰减尺度），
     看 rho_max、phi 全局 std/range、1-ring 残差 max/median 的响应幅度
  C) 把 m0 加回来（m(dev)=m0+a*dev），联合优化 (m0,a,W)，看能否超过方案 C

副产物：若 rho 不敏感但 phi 方差/残差敏感 => 偏转函数的真正价值在"均匀性"而非"覆盖半径"。
"""
import numpy as np, math, json, time
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize
import improve_corrected as ic

OUT = "E:/毕业论文/算法"
n = 27

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
def rot_axis(k,th):
    k=normalize(k); c=math.cos(th); s=math.sin(th)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

# ---------- geometry (same as optimize_theta_joint.py) ----------
P0,fi,I,J,isedge=ic.gen_meta(n)
C,dev,is_vertex,_,min_bary,Cs=ic.build_fields(P0,fi)
N=len(P0); V=ic.V; faces=ic.faces
u_rad=np.array([normalize(C[k]-np.dot(C[k],P0[k])*P0[k]) for k in range(N)])
d_e=np.zeros(N); u_tan=np.zeros((N,3))
for k in range(N):
    f=faces[fi[k]]; A,B,Cc=V[f[0]],V[f[1]],V[f[2]]
    best=None; bd=1e9
    for (X,Y) in [(A,B),(B,Cc),(Cc,A)]:
        ne=normalize(np.cross(X,Y))
        dd=math.degrees(math.asin(min(1.0,abs(np.dot(P0[k],ne)))))
        if dd<bd: bd=dd; best=(X,Y,ne)
    X,Y,ne=best; d_e[k]=bd
    ue=normalize(np.cross(P0[k],ne))
    s=ic.angle(P0[k],X)-ic.angle(P0[k],Y)
    u_tan[k]=(1.0 if s>=0 else -1.0)*ue

av=normalize(V[1]); af=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(av,2*math.pi/5),rot_axis(av,-2*math.pi/5),
      rot_axis(af,2*math.pi/3),rot_axis(af,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); key=tuple(np.round(M.ravel(),8))
    if key in seen: continue
    seen.add(key); G60.append(M)
    for g in gens: todo.append(g@M)
G120=G60+[-g for g in G60]
IDX=[cKDTree(P0).query((g@P0.T).T)[1] for g in G120]
Arad=np.zeros((N,3)); Atan=np.zeros((N,3))
for g,j in zip(G120,IDX):
    Arad+=(g.T@u_rad[j].T).T; Atan+=(g.T@u_tan[j].T).T
Arad/=len(G120); Atan/=len(G120)

print("=== A) Reynolds 后两个固定场的相对大小 ===")
aR=np.linalg.norm(Arad,axis=1); aT=np.linalg.norm(Atan,axis=1)
act=(~is_vertex)&(aR>1e-12)
print(f"  mean|Arad|={aR[act].mean():.6f}  mean|Atan|={aT[act].mean():.6f}  "
      f"比值 |Atan|/|Arad| = {aT[act].mean()/aR[act].mean():.4f}")
print(f"  中位数 cos(Arad,Atan) 夹角 = "
      f"{np.degrees(np.arccos(np.clip(np.sum(Arad[act]*Atan[act],1)/(aR[act]*aT[act]+1e-30),-1,1))).mean():.2f} deg")
print(f"  |Atan|<1e-6 的点数 = {(aT<1e-6).sum()} / {N}")

# ---------- metrics ----------
def circum_all(P):
    h=ConvexHull(P); t=h.simplices
    a0,b0,c0=P[t[:,0]],P[t[:,1]],P[t[:,2]]
    aa=np.arccos(np.clip(np.sum(a0*b0,1),-1,1))
    bb=np.arccos(np.clip(np.sum(b0*c0,1),-1,1))
    cc=np.arccos(np.clip(np.sum(c0*a0,1),-1,1))
    s=(aa+bb+cc)/2
    val=np.sin(s-aa)*np.sin(s-bb)*np.sin(s-cc)/np.sin(s)
    r=np.degrees(2*np.arctan(np.sqrt(np.clip(val,0,None))))
    return t,r

def metrics(P):
    t,r=circum_all(P)
    rmax=np.full(N,-np.inf); rmin=np.full(N,np.inf)
    np.maximum.at(rmax,t.ravel(),np.repeat(r,3))
    np.minimum.at(rmin,t.ravel(),np.repeat(r,3))
    res=rmax-rmin
    return dict(rho=float(r.max()),phi_std=float(r.std()),phi_range=float(r.ptp()),
                res_max=float(res.max()),res_median=float(np.median(res)))

def build_P(m, w):
    th=(math.pi/2)*np.clip(w,0,1)
    al=m*np.cos(th); be=m*np.sin(th)
    v=al[:,None]*Arad+be[:,None]*Atan
    mag=np.linalg.norm(v,axis=1)
    u=np.zeros_like(v); nz=mag>1e-12; u[nz]=v[nz]/mag[nz,None]
    t=np.radians(mag)[:,None]
    P=np.cos(t)*P0+np.sin(t)*u
    P/=np.linalg.norm(P,axis=1,keepdims=True)
    P[is_vertex]=P0[is_vertex]
    return P

# ---------- B) 固定 a 扫描 W ----------
a_fix=0.0041742
m_fix=a_fix*dev
WLIST={
 'w=0 pure radial': np.zeros(N),
 'w=1 pure tangent': np.ones(N),
 'exp lam=0.5': np.exp(-d_e/0.5),
 'exp lam=1': np.exp(-d_e/1.0),
 'exp lam=3 (opt)': np.exp(-d_e/3.0558),
 'exp lam=10': np.exp(-d_e/10.0),
 'linear lam=5': np.clip(1-d_e/5.0,0,1),
 'linear lam=15': np.clip(1-d_e/15.0,0,1),
 'step d_e<3': (d_e<3.0).astype(float),
}
print("\n=== B) 固定 a=0.0041742，扫描 W（看各指标响应幅度）===")
print(f"  {'W':22s} {'rho':>10s} {'phi_std':>10s} {'phi_range':>11s} {'res_max':>10s} {'res_med':>10s}")
resB={}
for name,w in WLIST.items():
    mt=metrics(build_P(m_fix,w)); resB[name]=mt
    print(f"  {name:22s} {mt['rho']:10.6f} {mt['phi_std']:10.6f} {mt['phi_range']:11.6f} "
          f"{mt['res_max']:10.6f} {mt['res_median']:10.6f}")
for key in ['rho','phi_std','phi_range','res_max','res_median']:
    vals=[r[key] for r in resB.values()]
    print(f"    {key:12s} 跨度 = {max(vals)-min(vals):.6f}  (min={min(vals):.6f}, max={max(vals):.6f})")

# ---------- C) 加回 m0 的联合优化 ----------
print("\n=== C) m(dev)=m0+a*dev，联合优化 (m0,a,lam) [exp 族] ===")
def objC(z):
    m0,a,lam=np.exp(z)
    w=np.exp(-d_e/lam)
    return metrics(build_P(m0+a*dev,w))['rho']
best=None
for s0 in [[0.001,0.0042,3.0],[0.0001,0.0042,5.0],[0.002,0.0042,1.5]]:
    z0=np.log(np.array(s0,float))
    simp=np.array([z0]+[z0+np.eye(3)[i]*0.08 for i in range(3)])
    r=minimize(objC,z0,method='Nelder-Mead',
               options={'maxiter':400,'initial_simplex':simp,'xatol':1e-6,'fatol':1e-8})
    if best is None or r.fun<best[0]: best=(float(r.fun),np.exp(r.x).tolist())
m0,a,lam=best[1]
mtC=metrics(build_P(m0+a*dev,np.exp(-d_e/lam)))
print(f"  最优 m0={m0:.8f}  a={a:.8f}  lam={lam:.4f}  rho={best[0]:.6f}")
print(f"  完整指标: {mtC}")
print(f"  对照 方案C=1.496089  方案A=1.492573  无m0最优=1.496312")
print(f"  -> 加回 m0 后相对方案C改善 = {1.496089025393318-best[0]:+.6f} deg")

json.dump(dict(A=dict(mean_Arad=float(aR[act].mean()),mean_Atan=float(aT[act].mean()),
                      ratio=float(aT[act].mean()/aR[act].mean()),
                      atan_zero_points=int((aT<1e-6).sum())),
               B=resB,B_span={k:max(r[k] for r in resB.values())-min(r[k] for r in resB.values())
                              for k in ['rho','phi_std','phi_range','res_max','res_median']},
               C=dict(m0=m0,a=a,lam=lam,rho=best[0],metrics=mtC)),
          open(OUT+"/diagnose_theta_insensitivity.json","w"),indent=2)
print("\n[完成] diagnose_theta_insensitivity.json")
