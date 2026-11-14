"""
最终交付：两种严格对称修正的三张图 + 两版箭头对照
====================================================
  [C] 协变切向版 = 保留"边界感知"的对称算法（用户 12:15 拼图本意）
      切向乘关于棱中点的奇函数 => 镜像两侧反号 => 严格对称、无环绕、边缘仍沿棱滑动
      rho=1.496089, scale=1.0
  [A] 纯径向对称版 = Reynolds 投影原场，切向被抵消
      rho=1.492573, scale=1.5，但边缘不再沿棱滑动

输入：v_covariant_tangent.npy (C)、v_symmetric_final.npy (A)
输出：
  fig_final_C_2x2.png             C 版修正前/后 热力图 + 残差图
  fig_final_C_displacement.png    C 版正确位移箭头（边缘切向两侧反号，无环绕）
  fig_final_residual_3variants.png  P0 / 旧版bug / C 版 三方残差
  fig_final_A_vs_C_arrows.png     A 与 C 箭头并排（说明"严格对称"不唯一）
  plot_final_symmetric.json       定量
"""
import numpy as np, math, json, csv
from scipy.spatial import ConvexHull
import improve_corrected as ic
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

OUT="E:/毕业论文/算法"
rows=list(csv.DictReader(open(OUT+"/symmetric_points_n27.csv")))
P0=np.array([[float(r['x']),float(r['y']),float(r['z'])] for r in rows])
fi=np.array([int(r['face_idx']) for r in rows])
I=np.array([int(r['i']) for r in rows]); J=np.array([int(r['j']) for r in rows])
isedge=np.array([int(r['is_edge']) for r in rows],bool)
isv=np.array([int(r['is_vertex']) for r in rows],bool)
C,dev,iv,ue,mb,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,iv,ue,isedge,mb)
raw=json.load(open(OUT+"/corrected_coef_n27.json"))["raw"]
Pold=corr.apply(raw)

def build(v,s):
    P=np.empty_like(P0); mag=np.linalg.norm(v,axis=1)
    for k in range(len(P0)):
        if iv[k] or mag[k]<1e-12: P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],s*mag[k],normalize(v[k]))
    return P
vC=np.load(OUT+"/v_covariant_tangent.npy"); PC=build(vC,1.0)
vA=np.load(OUT+"/v_symmetric_final.npy");  PA=build(vA,1.5)

n=27
sel=np.where(fi==0)[0]
loc={(int(I[k]),int(J[k])):i for i,k in enumerate(sel)}
A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tri2=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])
mid=[normalize(B+Cc),normalize(A+Cc),normalize(A+B)]
tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
P2=to2d(P0[sel])

def analyze(P):
    Pf=P[sel]
    phi=np.array([math.degrees(ic.sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    cent=np.array([normalize(Pf[t].sum(axis=0)) for t in tris])
    ring=[[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    return phi,to2d(cent),np.array([max(a)-min(a) for a in ring])
def st(x):
    return dict(min=float(x.min()),max=float(x.max()),range=float(x.max()-x.min()),
                mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)))
def rg(P):
    h=ConvexHull(P)
    return math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))

phi0,cc0,r0=analyze(P0); phiC,ccC,rC=analyze(PC); _,_,rA=analyze(PA); _,_,rOld=analyze(Pold)
R=dict(phi_before=st(phi0),phi_C=st(phiC),phi_A=analyze(PA)[0].std(),
       res_before=st(r0),res_C=st(rC),res_A=st(rA),res_old=st(rOld),
       rho_before=rg(P0),rho_C=rg(PC),rho_A=rg(PA),rho_old=rg(Pold))
print("=== rho_max ===")
print(f"  P0={R['rho_before']:.6f}  旧版bug={R['rho_old']:.6f}  "
      f"C协变切向={R['rho_C']:.6f}  A纯径向={R['rho_A']:.6f}")
print("=== 热力图 phi ===")
for k in ['phi_before','phi_C']:
    s=R[k]; print(f"  {k}: range={s['range']:.6f} std={s['std']:.6f}")
print("=== 1-ring 残差 ===")
for k in ['res_before','res_old','res_C','res_A']:
    s=R[k]; print(f"  {k}: median={s['median']:.6f} max={s['max']:.6f} mean={s['mean']:.6f}")

# ---- 图1：C 版 2x2 ----
fig,ax=plt.subplots(2,2,figsize=(11,10))
pvmin=min(R['phi_before']['min'],R['phi_C']['min']); pvmax=max(R['phi_before']['max'],R['phi_C']['max'])
rvmax=max(R['res_before']['max'],R['res_C']['max'])
for a in ax.ravel():
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
for j,(cc,phi,s,t) in enumerate([(cc0,phi0,R['phi_before'],'BEFORE'),
                                  (ccC,phiC,R['phi_C'],'AFTER (covariant tangent, symmetric)')]):
    sc=ax[0,j].scatter(cc[:,0],cc[:,1],c=phi,s=11,cmap='coolwarm',vmin=pvmin,vmax=pvmax)
    ax[0,j].set_title(f'Heat map phi(t) - {t}\nrange={s["range"]:.6f} std={s["std"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[0,j],shrink=.8,label='phi(t), deg')
for j,(res,s,t) in enumerate([(r0,R['res_before'],'BEFORE'),(rC,R['res_C'],'AFTER (covariant tangent)')]):
    sc=ax[1,j].scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rvmax)
    ax[1,j].set_title(f'1-ring residual - {t}\nmedian={s["median"]:.6f} max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[1,j],shrink=.8,label='r_i, deg')
fig.suptitle(f'face 0, n=27 | CORRECT boundary-aware correction (variant C)\n'
             f'strictly 120-fold symmetric, edge tangent flips sign across each midpoint',fontsize=11)
fig.tight_layout(); fig.savefig(OUT+"/fig_final_C_2x2.png",dpi=145)
print("\n[出图] fig_final_C_2x2.png")

# ---- 图2：C 版箭头 ----
AMP=180.0
def arrow_panel(a,P,title):
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.3)
    for vv,mm in zip([A,B,Cc],mid):
        ln=to2d(np.array([vv,mm])); a.plot(ln[:,0],ln[:,1],'k--',lw=.6,alpha=.3)
    dd=to2d(P[sel])-P2
    a.quiver(P2[:,0],P2[:,1],dd[:,0]*AMP,dd[:,1]*AMP,angles='xy',scale_units='xy',scale=1.0,
             color='tab:red',width=.0022,alpha=.8)
    a.set_aspect('equal'); a.axis('off'); a.set_title(title,fontsize=10)

fig2,ax2=plt.subplots(figsize=(8,7.2))
arrow_panel(ax2,PC,f'CORRECT boundary-aware field (variant C)\n'
                   f'arrows x{AMP:.0f} | rho_max={R["rho_C"]:.6f} deg\n'
                   f'edge tangent preserved, flips sign across midpoints => NO circulation')
fig2.tight_layout(); fig2.savefig(OUT+"/fig_final_C_displacement.png",dpi=145)
print("[出图] fig_final_C_displacement.png")

# ---- 图3：三方残差 ----
rv=max(R['res_before']['max'],R['res_C']['max'],R['res_old']['max'])
fig3,ax3=plt.subplots(1,3,figsize=(15,4.8))
for a,(res,t) in zip(ax3,[(r0,'P0 baseline'),(rOld,'OLD code (asymmetric bug)'),(rC,'NEW variant C')]):
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
    sc=a.scatter(P2[:,0],P2[:,1],c=res,s=14,cmap='viridis',vmin=0,vmax=rv)
    a.set_title(f'{t}\nmedian={np.median(res):.5f} max={res.max():.5f} deg',fontsize=10)
    plt.colorbar(sc,ax=a,shrink=.8,label='r_i, deg')
fig3.suptitle('1-ring residual: baseline vs buggy implementation vs correct symmetric correction',fontsize=11)
fig3.tight_layout(); fig3.savefig(OUT+"/fig_final_residual_3variants.png",dpi=145)
print("[出图] fig_final_residual_3variants.png")

# ---- 图4：A vs C 箭头 ----
fig4,ax4=plt.subplots(1,2,figsize=(13,6))
arrow_panel(ax4[0],PA,f'[A] pure-radial symmetric  rho={R["rho_A"]:.6f}\n'
                       f'edge tangent fully cancelled (no boundary-aware glide)')
arrow_panel(ax4[1],PC,f'[C] covariant-tangent symmetric  rho={R["rho_C"]:.6f}\n'
                       f'edge tangent preserved, mirror-symmetric (matches your 6-fold figure)')
fig4.suptitle('Both are strictly 120-fold symmetric; only C keeps the boundary-aware character',fontsize=11)
fig4.tight_layout(); fig4.savefig(OUT+"/fig_final_A_vs_C_arrows.png",dpi=145)
print("[出图] fig_final_A_vs_C_arrows.png")

json.dump({k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in R.items()},
          open(OUT+"/plot_final_symmetric.json","w"),indent=2)
print("\n[完成] plot_final_symmetric.json")
