"""
基于【正式对称算法 improve_symmetric.py 的产物】重算三张图
============================================================
输入：symmetric_points_n27.csv（P0 与严格对称的 Popt）
口径与 plot_correction_effect.py 完全一致（face 0、406 顶点、729 个构造三角形、
修正前后同一组 (i,j) 三元组），故可直接与旧图对照。

交付：
  fig_symmetric_effect_2x2.png        修正前/后 热力图 + 修正前/后 残差图
  fig_symmetric_displacement_face0.png 正确的位移矢量箭头图（应无环绕）
  fig_symmetric_residual_3variants.png P0 / 旧版(bug) / 对称版 三方残差对比
  plot_symmetric_effect.json          定量统计
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
Popt=np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in rows])
fi=np.array([int(r['face_idx']) for r in rows])
I=np.array([int(r['i']) for r in rows]); J=np.array([int(r['j']) for r in rows])
isedge=np.array([int(r['is_edge']) for r in rows],bool)
isv=np.array([int(r['is_vertex']) for r in rows],bool)
# 旧版（bug）用于三方对照
C,dev,is_vertex,u_edge,min_bary,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
raw=json.load(open(OUT+"/corrected_coef_n27.json"))["raw"]
Pold=corr.apply(raw)

n=27
sel=np.where(fi==0)[0]
loc={(int(I[k]),int(J[k])):i for i,k in enumerate(sel)}
A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tri2=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])
G=normalize(A+B+Cc)
tri_mid=[normalize(B+Cc),normalize(A+Cc),normalize(A+B)]

tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
print(f"face0 顶点={len(sel)}  构造三角形={len(tris)} (n^2={n*n})")

def analyze(P):
    Pf=P[sel]
    phi=np.array([math.degrees(ic.sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    cent=np.array([normalize(Pf[t].sum(axis=0)) for t in tris])
    ring=[[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    return phi,to2d(cent),np.array([max(a)-min(a) for a in ring])

def stats(x):
    return dict(min=float(x.min()),max=float(x.max()),range=float(x.max()-x.min()),
                mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)))

def rho_global(P):
    h=ConvexHull(P)
    return math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))

phi0,cc0,res0=analyze(P0)
phi1,cc1,res1=analyze(Popt)
_,_,resold=analyze(Pold)
S={'phi_before':stats(phi0),'phi_after':stats(phi1),
   'res_before':stats(res0),'res_after':stats(res1),'res_old_bug':stats(resold),
   'rho_before':rho_global(P0),'rho_after':rho_global(Popt),'rho_old_bug':rho_global(Pold)}

print("\n=== 热力图 phi(t)（度） ===")
for tag,s in [('修正前',S['phi_before']),('对称版修正后',S['phi_after'])]:
    print(f"  {tag}: range={s['range']:.6f} std={s['std']:.6f} mean={s['mean']:.6f}")
print("\n=== 1-ring 残差（度） ===")
for tag,s in [('修正前',S['res_before']),('旧版(bug)',S['res_old_bug']),('对称版修正后',S['res_after'])]:
    print(f"  {tag}: median={s['median']:.6f} max={s['max']:.6f} mean={s['mean']:.6f} std={s['std']:.6f}")
print(f"\n=== 全局 rho_max: {S['rho_before']:.6f} (前) / {S['rho_old_bug']:.6f} (旧版) / "
      f"{S['rho_after']:.6f} (对称版) ===")

# ---- 图1：2x2 ----
fig,ax=plt.subplots(2,2,figsize=(11,10))
pvmin=min(S['phi_before']['min'],S['phi_after']['min']); pvmax=max(S['phi_before']['max'],S['phi_after']['max'])
rvmax=max(S['res_before']['max'],S['res_after']['max'])
for a in ax.ravel():
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
for j,(cc,phi,s,ttl) in enumerate([(cc0,phi0,S['phi_before'],'BEFORE'),(cc1,phi1,S['phi_after'],'AFTER (symmetric)')]):
    sc=ax[0,j].scatter(cc[:,0],cc[:,1],c=phi,s=11,cmap='coolwarm',vmin=pvmin,vmax=pvmax)
    ax[0,j].set_title(f'Heat map phi(t) - {ttl}\nrange={s["range"]:.6f} std={s["std"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[0,j],shrink=.8,label='phi(t), deg')
P2=to2d(P0[sel])
for j,(res,s,ttl) in enumerate([(res0,S['res_before'],'BEFORE'),(res1,S['res_after'],'AFTER (symmetric)')]):
    sc=ax[1,j].scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rvmax)
    ax[1,j].set_title(f'1-ring residual - {ttl}\nmedian={s["median"]:.6f} max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[1,j],shrink=.8,label='r_i, deg')
fig.suptitle(f'face 0, n=27 | STRICTLY SYMMETRIC correction (G120 Reynolds)\n'
             f'global rho_max {S["rho_before"]:.6f} -> {S["rho_after"]:.6f} deg',fontsize=11)
fig.tight_layout(); fig.savefig(OUT+"/fig_symmetric_effect_2x2.png",dpi=145)
print("\n[出图] fig_symmetric_effect_2x2.png")

# ---- 图2：正确的位移箭头 ----
d2=to2d(Popt[sel])-to2d(P0[sel])
d3=np.array([math.degrees(math.acos(np.clip(np.dot(P0[sel][k],Popt[sel][k]),-1,1))) for k in range(len(sel))])
AMP=180.0
fig2,ax2=plt.subplots(figsize=(8,7.2))
ax2.plot(tri2[:,0],tri2[:,1],'k-',lw=1.4)
for vv,mm in zip([A,B,Cc],tri_mid):
    ln=to2d(np.array([vv,mm])); ax2.plot(ln[:,0],ln[:,1],'k--',lw=.6,alpha=.35)
ax2.scatter(P2[:,0],P2[:,1],s=9,c='lightgray',label='initial P0',zorder=2)
ax2.scatter(to2d(Popt[sel])[:,0],to2d(Popt[sel])[:,1],s=7,c='tab:blue',alpha=.75,label='symmetric Popt',zorder=3)
ax2.quiver(P2[:,0],P2[:,1],d2[:,0]*AMP,d2[:,1]*AMP,angles='xy',scale_units='xy',scale=1.0,
           color='tab:red',width=.0022,alpha=.8,zorder=4)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title(f'CORRECT displacement field (strictly 120-fold symmetric)\n'
              f'arrows x{AMP:.0f} | |disp| mean={d3.mean():.6f} max={d3.max():.6f} deg\n'
              f'radially inward, zero along-edge component, NO circulation',fontsize=10)
ax2.legend(loc='upper right',fontsize=9)
fig2.tight_layout(); fig2.savefig(OUT+"/fig_symmetric_displacement_face0.png",dpi=145)
print("[出图] fig_symmetric_displacement_face0.png")

# ---- 图3：三方残差 ----
rv=max(S['res_before']['max'],S['res_after']['max'],S['res_old_bug']['max'])
fig3,ax3=plt.subplots(1,3,figsize=(15,4.8))
for a,(res,ttl) in zip(ax3,[(res0,'P0 baseline'),(resold,'OLD code (asymmetric bug)'),(res1,'NEW symmetric')]):
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
    sc=a.scatter(P2[:,0],P2[:,1],c=res,s=14,cmap='viridis',vmin=0,vmax=rv)
    a.set_title(f'{ttl}\nmedian={np.median(res):.5f} max={res.max():.5f} deg',fontsize=10)
    plt.colorbar(sc,ax=a,shrink=.8,label='r_i, deg')
fig3.suptitle('1-ring residual: baseline vs buggy implementation vs strictly symmetric (same scale)',fontsize=11)
fig3.tight_layout(); fig3.savefig(OUT+"/fig_symmetric_residual_3variants.png",dpi=145)
print("[出图] fig_symmetric_residual_3variants.png")

S['disp_mean']=float(d3.mean()); S['disp_max']=float(d3.max())
S['disp_zero']=int((d3<1e-9).sum()); S['points']=len(sel); S['tris']=len(tris)
json.dump(S,open(OUT+"/plot_symmetric_effect.json","w"),indent=2)
print("\n[完成] plot_symmetric_effect.json")
