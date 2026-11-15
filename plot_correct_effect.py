"""
按【正确外接圆半径】重画位置拟合优化的可视化
==============================================
数据：baseline_P0_n27.csv / correct_points_L.csv / correct_points_J.csv
半径：covering_radius.sph_circumradius（向量几何定义），此前 fig_joint_*.png
      全部基于错误公式 2r，已作废。

输出：
  fig_correct_2x2.png             P0 / 方案J 热力图 + 残差图
  fig_correct_arrows_compare.png  方案L / 方案J 短箭头并排
  fig_correct_arrows_after.png    方案J 等长箭头 + 幅度颜色
  fig_correct_residual_3way.png   P0 / L / J 残差三方对照
  plot_correct_effect.json        定量
"""
import numpy as np, math, json, csv, sys
import improve_corrected as ic
from covering_radius import sph_circumradius, rho as rho_global
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "E:/毕业论文/算法"; n = 27
def normalize(v):
    v = np.asarray(v, float); s = np.linalg.norm(v); return v/s if s > 1e-15 else v*0.0

def load(path):
    rows = list(csv.DictReader(open(path)))
    P0 = np.array([[float(r['x']),float(r['y']),float(r['z'])] for r in rows])
    P  = np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in rows])
    fi = np.array([int(r['face_idx']) for r in rows])
    I  = np.array([int(r['i']) for r in rows]); J = np.array([int(r['j']) for r in rows])
    return P0, P, fi, I, J

P0, PJ, fi, I, J = load(f"{OUT}/correct_points_J.csv")
_, PL, _, _, _ = load(f"{OUT}/correct_points_L.csv")
_rows = list(csv.DictReader(open(f"{OUT}/baseline_P0_n27.csv")))
Pbase = np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in _rows])
assert np.abs(P0-Pbase).max() < 1e-12

V = ic.V; A,B,Cc = V[0],V[4],V[8]
e1 = normalize(B-A); nn = normalize(np.cross(B-A,Cc-A)); e2 = normalize(np.cross(nn,e1))
def to2d(X):
    Y = np.atleast_2d(X)-A; return np.c_[Y@e1, Y@e2]
tri2 = np.array([*to2d(np.array([A,B,Cc])), to2d(A)[0]])
grid = float(np.linalg.norm(to2d(B)-to2d(A)))/n

sel = np.where(fi==0)[0]
loc = {(int(I[k]),int(J[k])):h for h,k in enumerate(sel)}
tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris = np.array(tris)
P2 = to2d(P0[sel])
print(f"face0 格点={len(sel)} 构造三角形={len(tris)} 格距={grid:.6f}")

def analyze(P):
    Pf = P[sel]
    phi = np.array([math.degrees(sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    cent = np.array([normalize(Pf[t].sum(axis=0)) for t in tris])
    ring = [[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    res = np.array([max(a)-min(a) for a in ring])
    return phi, to2d(cent), res

def st(x):
    return dict(min=float(x.min()),max=float(x.max()),range=float(np.ptp(x)),
                mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)))

phi0,cc0,res0 = analyze(P0)
phiL,ccL,resL = analyze(PL)
phiJ,ccJ,resJ = analyze(PJ)
r0,rL,rJ = rho_global(P0), rho_global(PL), rho_global(PJ)
print(f"全局 rho: P0={r0:.6f}  L={rL:.6f}  J={rJ:.6f}")

mid = [normalize(B+Cc), normalize(A+Cc), normalize(A+B)]
def base_panel(ax):
    ax.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2)
    for vv,mm in zip([A,B,Cc],mid):
        ln = to2d(np.array([vv,mm])); ax.plot(ln[:,0],ln[:,1],'k--',lw=.5,alpha=.28)
    ax.scatter(P2[:,0],P2[:,1],s=3.5,c='0.80',edgecolors='none',zorder=2)
    ax.set_aspect('equal'); ax.axis('off')

# ---- 图1: 2x2 ----
fig,ax = plt.subplots(2,2,figsize=(11.5,10))
pvmin = min(phi0.min(),phiJ.min()); pvmax = max(phi0.max(),phiJ.max())
rvmax = max(res0.max(),resJ.max())
for a in ax.ravel(): base_panel(a)
for j,(cc,phi,s,t) in enumerate([(cc0,phi0,st(phi0),'BEFORE (P0)'),
                                  (ccJ,phiJ,st(phiJ),'AFTER (scheme J)')]):
    sc = ax[0,j].scatter(cc[:,0],cc[:,1],c=phi,s=11,cmap='coolwarm',vmin=pvmin,vmax=pvmax,zorder=3)
    ax[0,j].set_title(f'Heat map phi(t) - {t}\nrange={s["range"]:.6f}  std={s["std"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[0,j],shrink=.8,label='phi(t), deg')
for j,(res,s,t) in enumerate([(res0,st(res0),'BEFORE (P0)'),(resJ,st(resJ),'AFTER (scheme J)')]):
    sc = ax[1,j].scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rvmax,zorder=3)
    ax[1,j].set_title(f'1-ring residual - {t}\nmedian={s["median"]:.6f}  max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[1,j],shrink=.8,label='r_i, deg')
fig.suptitle(f'CORRECT circumradius | face 0, n=27 | rho_max {r0:.6f} -> {rJ:.6f} deg\n'
             f'phi std {st(phi0)["std"]:.6f} -> {st(phiJ)["std"]:.6f}',fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_correct_2x2.png",dpi=150)

# ---- 箭头 ----
dL = to2d(PL[sel])-P2; dJ = to2d(PJ[sel])-P2
mL,mJ = np.linalg.norm(dL,axis=1),np.linalg.norm(dJ,axis=1)
def ang3(P):
    return np.degrees(np.arccos(np.clip(np.sum(P0[sel]*P[sel],axis=1),-1,1)))
aL,aJ = ang3(PL),ang3(PJ)
AMP = 0.42*grid/max(mL.max(),mJ.max())
KW = dict(angles='xy',scale_units='xy',scale=1.0,width=.0016,
          headwidth=3,headlength=4,alpha=.88,zorder=4)
print(f"箭头 AMP={AMP:.2f}（旧错误口径图 AMP=6.10），最长 {0.42:.2f} 格距")

fig2,ax2 = plt.subplots(1,2,figsize=(13.5,6.4))
for a,(d,mag,t) in zip(ax2,[(dL,aL,f'[L] linear displacement  rho={rL:.6f} deg\n'
                                    f'm(dev)~m0+a*dev, |disp| mean={aL.mean():.4f} deg'),
                             (dJ,aJ,f'[J] joint angle+magnitude  rho={rJ:.6f} deg\n'
                                    f'|disp| mean={aJ.mean():.4f} deg')]):
    base_panel(a)
    a.quiver(P2[:,0],P2[:,1],d[:,0]*AMP,d[:,1]*AMP,color='tab:red',**KW)
    a.set_title(t,fontsize=9.5,pad=8)
fig2.suptitle(f'CORRECT circumradius | short arrows x{AMP:.1f}, longest = 0.42 grid spacing | '
              f'both panels share ONE scale',fontsize=10.5,y=.985)
fig2.tight_layout(rect=[0,0,1,.86]); fig2.savefig(f"{OUT}/fig_correct_arrows_compare.png",dpi=150)

fig3,ax3 = plt.subplots(figsize=(8.6,7.6))
base_panel(ax3)
nz = mJ>1e-12
U = np.zeros_like(dJ); U[nz] = dJ[nz]/mJ[nz,None]; U *= 0.36*grid
sc = ax3.quiver(P2[:,0],P2[:,1],U[:,0],U[:,1],aJ,cmap='viridis',
                angles='xy',scale_units='xy',scale=1.0,width=.0018,
                headwidth=3.5,headlength=4.5,alpha=.92,zorder=4)
plt.colorbar(sc,ax=ax3,shrink=.78,label='|disp| (deg)')
ax3.set_title(f'[J] unit-length direction arrows (0.36 grid spacing)\n'
              f'colour = true displacement | rho={rJ:.6f} deg',fontsize=10)
fig3.tight_layout(); fig3.savefig(f"{OUT}/fig_correct_arrows_after.png",dpi=150)

fig4,ax4 = plt.subplots(1,3,figsize=(15,4.9))
rv = max(res0.max(),resL.max(),resJ.max())
for a,(res,s,t) in zip(ax4,[(res0,st(res0),'P0 baseline'),(resL,st(resL),'scheme L'),
                             (resJ,st(resJ),'scheme J')]):
    base_panel(a)
    sc = a.scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rv,zorder=3)
    a.set_title(f'{t}\nmedian={s["median"]:.6f}  max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=a,shrink=.8,label='r_i, deg')
fig4.suptitle('CORRECT circumradius | 1-ring residual: P0 / L / J',fontsize=11)
fig4.tight_layout(); fig4.savefig(f"{OUT}/fig_correct_residual_3way.png",dpi=150)

json.dump(dict(radius_formula="GEOM: R=arccos(|n.a|)",
               rho=dict(P0=r0,L=rL,J=rJ),
               phi=dict(P0=st(phi0),L=st(phiL),J=st(phiJ)),
               residual=dict(P0=st(res0),L=st(resL),J=st(resJ)),
               arrows=dict(AMP=float(AMP),grid=float(grid),longest_in_grid=0.42,
                           disp_mean_L=float(aL.mean()),disp_max_L=float(aL.max()),
                           disp_mean_J=float(aJ.mean()),disp_max_J=float(aJ.max())),
               note="fig_joint_*.png / fig_final_C_*.png / fig_correction_*.png are VOID "
                    "(computed with the wrong 2r formula)."),
          open(f"{OUT}/plot_correct_effect.json","w"),indent=2)
print("[出图] fig_correct_2x2.png / fig_correct_arrows_compare.png / "
      "fig_correct_arrows_after.png / fig_correct_residual_3way.png")
print("[交付] plot_correct_effect.json")
