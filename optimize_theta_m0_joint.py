"""
最终联合优化：m(dev)=m0+a*dev 与偏转函数 W 的完全耦合
=====================================================
诊断发现：fixed W + 自由 (m0,a) 无法突破，而 (m0,a,W) 联合可到 rho=1.491822。
本脚本把这一耦合坐实，并对全部 7 个 W 族做联合优化，给出最终推荐解 + 完整门禁。

对照（回答"m0 的增益是否来自与 W 的耦合"）：
  R1  W=0   固定纯径向，只优化 (m0,a)
  R2  W=exp(-d_e/3.0558) 固定，只优化 (m0,a)
  R3  W=1   固定纯切向，只优化 (m0,a)
  J*  7 个族各自联合优化 (m0,a,shape)

门禁：点数 7292 / rho<=1.5 / 严格 120 阶对称 0/120 / 镜像 0 / 三棱无环流
"""
import numpy as np, math, json, csv, time
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize
import improve_corrected as ic

OUT="E:/毕业论文/算法"; n=27
def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
def rot_axis(k,th):
    k=normalize(k); c=math.cos(th); s=math.sin(th)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

# ---------- geometry ----------
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

def rho_fast(P):
    h=ConvexHull(P); t=h.simplices
    a0,b0,c0=P[t[:,0]],P[t[:,1]],P[t[:,2]]
    aa=np.arccos(np.clip(np.sum(a0*b0,1),-1,1))
    bb=np.arccos(np.clip(np.sum(b0*c0,1),-1,1))
    cc=np.arccos(np.clip(np.sum(c0*a0,1),-1,1))
    s=(aa+bb+cc)/2
    val=np.sin(s-aa)*np.sin(s-bb)*np.sin(s-cc)/np.sin(s)
    return math.degrees(float((2*np.arctan(np.sqrt(np.clip(val,0,None)))).max()))

def build_P(m0,a,w,v_override=None):
    m=m0+a*dev
    th=(math.pi/2)*np.clip(w,0,1)
    al=m*np.cos(th); be=m*np.sin(th)
    v=al[:,None]*Arad+be[:,None]*Atan if v_override is None else v_override
    mag=np.linalg.norm(v,axis=1)
    u=np.zeros_like(v); nz=mag>1e-12; u[nz]=v[nz]/mag[nz,None]
    t=np.radians(mag)[:,None]
    P=np.cos(t)*P0+np.sin(t)*u
    P/=np.linalg.norm(P,axis=1,keepdims=True)
    P[is_vertex]=P0[is_vertex]
    return P,v

def W(kind,x,q):
    if kind=='linear':     lam,=q; return np.clip(1-x/lam,0,1)
    if kind=='power':      lam,p=q; return np.clip(1-x/lam,0,1)**p
    if kind=='exp':        lam,=q; return np.exp(-x/lam)
    if kind=='stretched':  lam,p=q; return np.exp(-(x/lam)**p)
    if kind=='cos':        lam,=q; return np.cos((math.pi/2)*np.clip(x/lam,0,1))
    if kind=='linear_dev': lam,=q; return np.clip(x/lam,0,1)
    if kind=='power_dev':  lam,p=q; return np.clip(x/lam,0,1)**p
    raise ValueError(kind)

# ---------- R1/R2/R3: fixed W ----------
print("=== 对照组：固定 W，只优化 (m0,a) ===")
fixed_res={}
for name,w in [('R1_W0_radial',np.zeros(N)),
               ('R2_W_exp3',np.exp(-d_e/3.0558)),
               ('R3_W1_tangent',np.ones(N))]:
    def objf(z,w=w):
        m0,a=np.exp(z); return rho_fast(build_P(m0,a,w)[0])
    best=None
    for s0 in [[0.001,0.0042],[0.01,0.004],[0.04,0.003]]:
        z0=np.log(np.array(s0,float))
        simp=np.array([z0]+[z0+np.eye(2)[i]*0.08 for i in range(2)])
        r=minimize(objf,z0,method='Nelder-Mead',
                   options={'maxiter':350,'initial_simplex':simp,'xatol':1e-6,'fatol':1e-8})
        if best is None or r.fun<best[0]: best=(float(r.fun),np.exp(r.x).tolist())
    fixed_res[name]=dict(rho=best[0],m0=best[1][0],a=best[1][1])
    print(f"  {name:18s} rho={best[0]:.6f}  m0={best[1][0]:.7f}  a={best[1][1]:.7f}")

# ---------- J*: 7 族联合优化 ----------
KINDS=['linear','power','exp','stretched','cos','linear_dev','power_dev']
STARTS={
 'linear':     [[0.001,0.0042,5],[0.02,0.0035,3],[0.04,0.003,8]],
 'power':      [[0.001,0.0042,8,1],[0.02,0.0035,8,2],[0.04,0.003,5,0.7]],
 'exp':        [[0.001,0.0042,3],[0.02,0.0035,2],[0.04,0.003,5]],
 'stretched':  [[0.001,0.0042,5,1.5],[0.02,0.0035,8,2],[0.04,0.003,3,1]],
 'cos':        [[0.001,0.0042,8],[0.02,0.0035,15],[0.04,0.003,5]],
 'linear_dev': [[0.001,0.0042,20],[0.02,0.0035,37],[0.04,0.003,25]],
 'power_dev':  [[0.001,0.0042,25,1],[0.02,0.0035,37,2],[0.04,0.003,20,0.7]],
}
print("\n=== 联合优化 (m0,a,shape) ===")
joint={}
for kind in KINDS:
    x=dev if kind.endswith('_dev') else d_e
    def objj(z,kind=kind,x=x):
        q=np.exp(z); m0,a=q[0],q[1]; shape=q[2:]
        return rho_fast(build_P(m0,a,W(kind,x,shape))[0])
    best=None
    for s0 in STARTS[kind]:
        z0=np.log(np.array(s0,float)); k=len(z0)
        simp=np.array([z0]+[z0+np.eye(k)[i]*0.08 for i in range(k)])
        r=minimize(objj,z0,method='Nelder-Mead',
                   options={'maxiter':500,'initial_simplex':simp,'xatol':1e-6,'fatol':1e-8})
        if best is None or r.fun<best[0]: best=(float(r.fun),np.exp(r.x).tolist(),int(r.nfev))
    q=best[1]
    joint[kind]=dict(rho=best[0],m0=q[0],a=q[1],shape=q[2:],nfev=best[2])
    print(f"  {kind:12s} rho={best[0]:.6f}  m0={q[0]:.7f}  a={q[1]:.7f}  "
          f"shape={np.round(q[2:],4)}  nfev={best[2]}")

# ---------- 最优解：完整 Reynolds 复核 + 门禁 ----------
bk=min(joint,key=lambda k:joint[k]['rho'])
q=joint[bk]; x=dev if bk.endswith('_dev') else d_e
w=W(bk,x,q['shape']); m0,a=q['m0'],q['a']
th=(math.pi/2)*np.clip(w,0,1); m=m0+a*dev
d0=(m*np.cos(th))[:,None]*u_rad+(m*np.sin(th))[:,None]*u_tan
acc=np.zeros((N,3))
for g,j in zip(G120,IDX): acc+=(g.T@d0[j].T).T
v_full=acc/len(G120)
P_fast,_=build_P(m0,a,w)
P_full,_=build_P(m0,a,w,v_override=v_full)
r_fast,r_full=rho_fast(P_fast),rho_fast(P_full)
print(f"\n[复核] 最优族={bk}  快速 rho={r_fast:.9f}  完整 Reynolds rho={r_full:.9f}  "
      f"点集差={np.abs(P_fast-P_full).max():.3e}")
assert abs(r_fast-r_full)<1e-8
P=P_full; r=r_full

def strict_sym(P,tol=1e-9):
    t=cKDTree(P); bad=0; worst=0.0
    for g in G120:
        d,j=t.query((g@P.T).T)
        if len(np.unique(j))!=len(P) or d.max()>tol: bad+=1
        worst=max(worst,float(d.max()))
    return bad,math.degrees(worst)
def mirror_median(P):
    A,B,Cc=V[0],V[4],V[8]; G=normalize(A+B+Cc)
    nA=normalize(np.cross(G,A)); S=np.eye(3)-2*np.outer(nA,nA)
    sel=np.where(fi==0)[0]; base=P0[sel]
    jj=cKDTree(base).query((S@base.T).T)[1]
    def td(Z):
        d=Z[sel]-base; return d-np.sum(d*base,axis=1)[:,None]*base
    d=td(P); Sd=(S@d.T).T
    num=np.sum(Sd*d[jj],axis=1)
    den=np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[jj],axis=1)+1e-30
    return float(np.median(np.degrees(np.arccos(np.clip(num/den,-1,1)))))
def edge_signed(P):
    sel=np.where(fi==0)[0]; base=P0[sel]
    d=P[sel]-base; d=d-np.sum(d*base,axis=1)[:,None]*base
    A,B,Cc=V[0],V[4],V[8]; out=[]
    for cond,u in [(lambda i,j:j==0,normalize(B-A)),(lambda i,j:j==i,normalize(Cc-A)),
                   (lambda i,j:i==n,normalize(Cc-B))]:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        pr=np.array([np.dot(d[h],u) for h in ks])
        out.append([float(pr.mean()),float(np.abs(pr).mean())])
    return out
bad,worst=strict_sym(P); mir=mirror_median(P); ec=edge_signed(P)
disp=np.degrees(np.arccos(np.clip(np.sum(P0*P,axis=1),-1,1)))
print(f"\n[门禁] 点数={len(P)} rho={r:.6f} 对称失败={bad}/120 偏差={worst:.9f} "
      f"镜像median={mir:.6f} 位移均值={disp.mean():.6f} 最大={disp.max():.6f}")
print(f"       三棱 (符号均值, |沿棱|) = {np.round(ec,9)}")
assert len(P)==7292 and bad==0 and r<=1.5 and mir<1e-3
assert max(abs(e[0]) for e in ec)<1e-6
print("[门禁全部通过]")

# ---------- 交付 ----------
np.save(OUT+"/v_theta_m0_joint_best.npy",v_full)
coef=dict(variant="joint_m0_theta_C",family=bk,xvar='dev' if bk.endswith('_dev') else 'd_e',
          m0=m0,a=a,shape=q['shape'],m_formula="m(dev)=m0+a*dev",
          theta_formula="theta=(pi/2)*W(x)",rho_max=r,
          strict_sym_bad=bad,sym_worst_deg=worst,mirror_median_deg=mir,
          edge_signed_mean=[e[0] for e in ec],edge_abs_mean=[e[1] for e in ec],
          disp_mean_deg=float(disp.mean()),disp_max_deg=float(disp.max()),
          baseline=dict(P0=1.502585272272, C_covariant_tangent=1.496089025393318,
                        A_pure_radial=1.492573, joint_no_m0=1.496311681),
          note="Joint optimization of (m0,a,W) breaks the rho plateau; m0 is reactivated "
               "through coupling with the deflection width.")
json.dump(coef,open(OUT+"/theta_m0_joint_coef_n27.json","w"),indent=2)
with open(OUT+"/theta_m0_joint_points_n27.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j",
                                   "dev_deg","d_e_deg","m_deg","theta_deg","disp_deg","is_vertex"])
    for k in range(N):
        wr.writerow([P0[k,0],P0[k,1],P0[k,2],P[k,0],P[k,1],P[k,2],int(fi[k]),int(I[k]),int(J[k]),
                     dev[k],d_e[k],m[k],math.degrees(th[k]),disp[k],int(is_vertex[k])])
json.dump(dict(fixed_W=fixed_res,joint=joint,best_kind=bk,rho=r,
               gate=dict(bad=bad,worst_deg=worst,mirror=mir,edges=ec,
                         disp_mean=float(disp.mean()),disp_max=float(disp.max()))),
          open(OUT+"/optimize_theta_m0_joint.json","w"),indent=2)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    xd=np.linspace(0,float(d_e.max()),200)
    for kind in KINDS:
        if kind.endswith('_dev'): continue
        ax[0].plot(xd,W(kind,xd,joint[kind]['shape']),lw=1.7,
                   label=f"{kind} rho={joint[kind]['rho']:.5f}")
    ax[0].set_xlabel('d_e (deg)'); ax[0].set_ylabel('w'); ax[0].set_title('Optimized W(d_e) with free m0')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    xv=np.linspace(0,float(dev.max()),200)
    ax[1].plot(xv,m0+a*xv,'r-',lw=2,label=f'opt: m0={m0:.4f}, a={a:.5f}')
    ax[1].plot(xv,0.001085+0.004132*xv,'b--',lw=1.6,label='scheme C: m0~0.0011, a=0.00413')
    ax[1].plot(xv,0.0041742*xv,'g:',lw=1.6,label='m0=0 joint: a=0.00417')
    ax[1].set_xlabel('dev (deg)'); ax[1].set_ylabel('m (deg)'); ax[1].set_title('Displacement magnitude m(dev)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT+"/fig_theta_m0_joint.png",dpi=140)
    print("[出图] fig_theta_m0_joint.png")
except Exception as e: print(f"[出图跳过] {e}")

print("\n================ 最终汇总 ================")
print(f"  P0=1.502585  方案C=1.496089  方案A=1.492573  (m0=0)联合=1.496312")
for k2,v2 in fixed_res.items(): print(f"  {k2:18s} rho={v2['rho']:.6f}")
print(f"  -> 联合最优 {bk}: rho={r:.6f}  m0={m0:.6f}  a={a:.7f}")
print(f"     相对方案C改善 = {1.496089025393318-r:.6f} deg")
print("[交付] theta_m0_joint_coef_n27.json / theta_m0_joint_points_n27.csv / v_theta_m0_joint_best.npy")
