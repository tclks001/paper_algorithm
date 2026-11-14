"""
最终严格验证 + 对照图
=====================
1) 严格多重集对称校验：对每个群元 g，要求 g·P 与 P 是【双射】且距离 < 1e-9
   （不用"最近邻最大距离"这种可能被假阳性骗过的弱指标）
2) 确认 Reynolds 对称化场确实对称，且 rho_max <= 1.5deg
3) 出对照图：face 0 上"代码实际方向" vs "镜像协变要求的方向"的偏差分布
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
N=len(P0)

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

def strict_sym(P,tol=1e-9):
    """严格：g·P 必须与 P 构成双射且逐点距离<tol"""
    tree=cKDTree(P); worst=0.0; bad=0; worst_g=None
    for gi,g in enumerate(G120):
        d,j=tree.query((g@P.T).T)
        bij = (len(np.unique(j))==len(P))
        if not bij or d.max()>tol: bad+=1
        if d.max()>worst: worst=float(d.max()); worst_g=gi
    return bad, math.degrees(worst), worst_g

# ---- 对称化场（重算，独立路径） ----
d0=m[:,None]*u_dir; d0[is_vertex]=0.0
tree0=cKDTree(P0)
acc=np.zeros((N,3))
for g in G120:
    d,j=tree0.query((g@P0.T).T)
    acc += (g.T @ d0[j].T).T
v=acc/len(G120)
mag=np.linalg.norm(v,axis=1)
P_sym=np.empty_like(P0)
for k in range(N):
    if is_vertex[k] or mag[k]<1e-12: P_sym[k]=P0[k]; continue
    P_sym[k]=ic.exp_map(P0[k],mag[k],normalize(v[k]))
P_old=corr.apply(raw)

print("=== 严格多重集对称校验 (120 群元, 要求双射+距离<1e-9) ===")
for name,P in [("P0 原始构造",P0),("P_old 原代码场",P_old),("P_sym 对称化场",P_sym)]:
    bad,w,wg=strict_sym(P)
    print(f"  {name:>16s}: 失败群元={bad:3d}/120  最大偏差={w:.9f}deg  "
          f"{'严格对称 OK' if bad==0 else 'NOT SYMMETRIC'}")
print(f"\n  rho_max: P0={ic.rho(P0):.6f}deg  P_old={ic.rho(P_old):.6f}deg  "
      f"P_sym={ic.rho(P_sym):.6f}deg   (<=1.5: {'YES' if ic.rho(P_sym)<=1.5 else 'NO'})")

# ---- tie 点 vs 零位移点 ----
Cs_arr=np.array([normalize(V[f[0]]+V[f[1]]+V[f[2]]) for f in faces])
D=P0@Cs_arr.T; srt=np.sort(D,axis=1); tie=(srt[:,-1]-srt[:,-2])<1e-9
zero=mag<1e-9
print(f"\n=== 对称轴点统计 ===")
print(f"  tie 点(到多个面心等距) = {tie.sum()}   [= 30棱 x 26内点 + 12顶点 = {30*26+12}]")
print(f"  对称化后零位移点       = {zero.sum()}")
print(f"  零位移 ∩ tie           = {(zero&tie).sum()}  (tie 点占零位移的 {(zero&tie).sum()/max(zero.sum(),1)*100:.1f}%)")

# ---- 对照图：face 0 上代码实际方向与镜像协变的偏差 ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f0=faces[0]; A,B,Cc=V[f0[0]],V[f0[1]],V[f0[2]]
    e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
    def to2d(X): Y=np.atleast_2d(X)-A; return np.c_[Y@e1, Y@e2]
    sel=np.where(fi==0)[0]
    P2=to2d(P0[sel]); sub=cKDTree(P0)
    # Mirror_A：过球心、面心 C0、顶点 A 的平面
    C0=normalize(A+B+Cc); nA=normalize(np.cross(C0,A))
    S=np.eye(3)-2*np.outer(nA,nA)
    d,j=sub.query((S@P0[sel].T).T)
    def vec2d(X): Y=np.atleast_2d(X); return np.c_[Y@e1, Y@e2]
    v_old=vec2d((m[sel,None]*u_dir[sel]))
    v_new=vec2d(v[sel])
    Sv=vec2d((S@(m[sel,None]*u_dir[sel]).T).T)
    vj_old=vec2d(m[j,None]*u_dir[j]); vj_new=vec2d(v[j])
    def ang(a,b):
        num=np.sum(a*b,axis=1); den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-30
        return np.degrees(np.arccos(np.clip(num/den,-1,1)))
    dev_old=ang(Sv,vj_old); dev_new=ang(vec2d((S@v[sel].T).T),vj_new)
    tri=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])
    fig,ax=plt.subplots(1,2,figsize=(13,5.6))
    for a,dd,vv,ttl in zip(ax,[dev_old,dev_new],[v_old,v_new],
                           ["ORIGINAL code field: mirror-covariance error",
                            "SYMMETRIZED (Reynolds G120): mirror-covariance error"]):
        a.plot(tri[:,0],tri[:,1],'k-',lw=1.4)
        sc=a.scatter(P2[:,0],P2[:,1],c=dd,s=13,cmap='coolwarm',vmin=0,vmax=180,zorder=3)
        sc2=a.quiver(P2[:,0],P2[:,1],vv[:,0]*30,vv[:,1]*30,angles='xy',scale_units='xy',
                     scale=1.0,color='k',alpha=.45,width=.0025,zorder=2)
        a.set_aspect('equal'); a.set_title(ttl+"\n(color: angle between S*u(p) and u(S*p), deg)",fontsize=9)
        a.set_xlabel('x'); a.set_ylabel('y'); plt.colorbar(sc,ax=a,shrink=.75)
    fig.suptitle(f"face 0 | original max err={dev_old.max():.1f}deg, symmetrized max err={dev_new.max():.1f}deg",
                 fontsize=10)
    fig.tight_layout(); fig.savefig("E:/毕业论文/算法/fig_symmetry_bug.png",dpi=140)
    print(f"\n[出图] fig_symmetry_bug.png  原版镜像偏差 max={dev_old.max():.2f}deg  "
          f"对称化后 max={dev_new.max():.2f}deg")
except Exception as e:
    print(f"[出图跳过] {e}")

json.dump({"strict_bad_P0":strict_sym(P0)[0],"strict_bad_Pold":strict_sym(P_old)[0],
           "strict_bad_Psym":strict_sym(P_sym)[0],
           "rho_Psym":ic.rho(P_sym),"rho_Pold":ic.rho(P_old),
           "tie_points":int(tie.sum()),"zero_disp":int(zero.sum())},
          open("E:/毕业论文/算法/verify_final_symmetry.json","w"),indent=2)
print("\n[完成] verify_final_symmetry.json")
