"""
边界感知修正前后：一个三角面内的热力图 / 1-ring 残差图 / 位移箭头图
====================================================================
口径（三者严格同源，保证修正前后可比）：
  - 只取 face 0，顶点 = gen_meta 中 fi==0 的点，按 (I,J) 建立索引
  - 构造网格小三角形（up/down 共 n^2=729 个），修正前后用同一组 (i,j) 三元组
  - 热力图：phi(t)=球面外接圆半径，画在三角形重心
  - 残差图：r_i = max-min(phi over 该点 1-ring 的构造三角形)，画在顶点
  - 箭头图：P0 -> Popt 的 2D 投影位移（放大显示）
另报全局 ConvexHull rho_max 作独立校验。

修正后点集 = 成稿 3.3.4 原版边界感知修正 corr.apply(raw)，rho=1.496089deg。
"""
import numpy as np, math, json, os
from scipy.spatial import ConvexHull
import improve_corrected as ic
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

n=27
P0,fi,I,J,isedge=ic.gen_meta(n)
C,dev,is_vertex,u_edge,min_bary,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
raw=json.load(open("E:/毕业论文/算法/corrected_coef_n27.json"))["raw"]
Popt=corr.apply(raw)

# ---- face 0 局部索引与 (I,J) 映射 ----
sel=np.where(fi==0)[0]
loc={ (int(I[k]),int(J[k])):i for i,k in enumerate(sel) }
P0f=P0[sel]; Pf=Popt[sel]

A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tri2=np.array([*to2d(np.array([A,B,Cc])), to2d(A)[0]])

# ---- 构造网格三角形（up / down，共 n^2） ----
tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i:
            tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
print(f"face0 顶点数={len(sel)}  构造三角形数={len(tris)} (应为 n^2={n*n})")

def analyze(P):
    phi=np.array([math.degrees(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]])) for t in tris])
    cent=np.array([normalize(P[t].sum(axis=0)) for t in tris])
    ring=[[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    res=np.array([max(a)-min(a) for a in ring])
    return phi,to2d(cent),res

phi0,cc0,res0=analyze(P0f)
phi1,cc1,res1=analyze(Pf)

def stats(x):
    return dict(min=float(x.min()),max=float(x.max()),range=float(x.max()-x.min()),
                mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)))
s_phi0,s_phi1=stats(phi0),stats(phi1)
s_res0,s_res1=stats(res0),stats(res1)

# 全局 Delaunay 校验
def rho_global(P):
    h=ConvexHull(P); return math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))
rg0,rg1=rho_global(P0),rho_global(Popt)

print("\n=== 热力图 phi(t) 统计（度） ===")
for tag,s in [("修正前",s_phi0),("修正后",s_phi1)]:
    print(f"  {tag}: min={s['min']:.6f} max={s['max']:.6f} range={s['range']:.6f} "
          f"std={s['std']:.6f} mean={s['mean']:.6f}")
print(f"  -> range {s_phi0['range']:.6f} -> {s_phi1['range']:.6f} "
      f"({(s_phi1['range']/s_phi0['range']-1)*100:+.2f}%), std {s_phi0['std']:.6f} -> {s_phi1['std']:.6f} "
      f"({(s_phi1['std']/s_phi0['std']-1)*100:+.2f}%)")
print("\n=== 1-ring 平衡残差 r_i 统计（度） ===")
for tag,s in [("修正前",s_res0),("修正后",s_res1)]:
    print(f"  {tag}: min={s['min']:.6f} median={s['median']:.6f} max={s['max']:.6f} "
          f"mean={s['mean']:.6f} std={s['std']:.6f}")
print(f"  -> median {s_res0['median']:.6f} -> {s_res1['median']:.6f} "
      f"({(s_res1['median']/s_res0['median']-1)*100:+.2f}%)")
print(f"\n=== 全局 Delaunay rho_max: {rg0:.6f} -> {rg1:.6f} (改善 {rg0-rg1:.6f}deg) ===")

# ---- 图 1：2x2 热力图 / 残差图，前后共用 colorbar 范围 ----
fig,ax=plt.subplots(2,2,figsize=(11,10))
pvmin=min(s_phi0['min'],s_phi1['min']); pvmax=max(s_phi0['max'],s_phi1['max'])
rvmin=0; rvmax=max(s_res0['max'],s_res1['max'])
for a in ax.ravel():
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
for j,(cc,phi,ttl) in enumerate([(cc0,phi0,'BEFORE correction'),(cc1,phi1,'AFTER correction')]):
    sc=ax[0,j].scatter(cc[:,0],cc[:,1],c=phi,s=11,cmap='coolwarm',vmin=pvmin,vmax=pvmax)
    ax[0,j].set_title(f'Heat map phi(t) - {ttl}\nrange={stats(phi)["range"]:.6f} std={stats(phi)["std"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[0,j],shrink=.8,label='phi(t), deg')
P2=to2d(P0f)
for j,(res,ttl) in enumerate([(res0,'BEFORE correction'),(res1,'AFTER correction')]):
    sc=ax[1,j].scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=rvmin,vmax=rvmax)
    ax[1,j].set_title(f'1-ring residual r_i - {ttl}\nmedian={np.median(res):.6f} max={res.max():.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[1,j],shrink=.8,label='r_i, deg')
fig.suptitle(f'face 0, n=27 | BEFORE vs AFTER boundary-aware correction\n'
             f'global rho_max {rg0:.6f} -> {rg1:.6f} deg',fontsize=11)
fig.tight_layout()
fig.savefig("E:/毕业论文/算法/fig_correction_effect_2x2.png",dpi=145)
print("\n[出图] fig_correction_effect_2x2.png")

# ---- 图 2：位移箭头图（整面） ----
d2=to2d(Pf)-to2d(P0f)
d3=np.array([math.degrees(math.acos(np.clip(np.dot(P0f[k],Pf[k]),-1,1))) for k in range(len(sel))])
AMP=180.0
fig2,ax2=plt.subplots(figsize=(8,7.2))
ax2.plot(tri2[:,0],tri2[:,1],'k-',lw=1.4)
# 三条中线（辅助定位）
G=normalize(A+B+Cc)
for vv in [A,B,Cc]:
    seg=to2d(np.array([vv,G])) if False else None
mid=[normalize(B+Cc),normalize(A+Cc),normalize(A+B)]
for vv,mm in zip([A,B,Cc],mid):
    ln=to2d(np.array([vv,mm])); ax2.plot(ln[:,0],ln[:,1],'k--',lw=.6,alpha=.35)
ax2.scatter(P2[:,0],P2[:,1],s=9,c='lightgray',edgecolors='none',label='initial P0',zorder=2)
ax2.scatter(to2d(Pf)[:,0],to2d(Pf)[:,1],s=7,c='tab:blue',alpha=.75,label='corrected Popt',zorder=3)
ax2.quiver(P2[:,0],P2[:,1],d2[:,0]*AMP,d2[:,1]*AMP,angles='xy',scale_units='xy',scale=1.0,
           color='tab:red',width=.0022,alpha=.8,zorder=4)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title(f'face 0 displacement field  P0 -> Popt  (arrows x{AMP:.0f})\n'
              f'|disp| mean={d3.mean():.6f} max={d3.max():.6f} deg, '
              f'zero={int((d3<1e-9).sum())} points',fontsize=10)
ax2.legend(loc='upper right',fontsize=9)
fig2.tight_layout(); fig2.savefig("E:/毕业论文/算法/fig_correction_displacement_face0.png",dpi=145)
print("[出图] fig_correction_displacement_face0.png")

json.dump(dict(phi_before=s_phi0,phi_after=s_phi1,res_before=s_res0,res_after=s_res1,
               rho_global_before=rg0,rho_global_after=rg1,
               disp_mean=float(d3.mean()),disp_max=float(d3.max()),
               disp_zero=int((d3<1e-9).sum()),tris=int(len(tris)),points=int(len(sel))),
          open("E:/毕业论文/算法/plot_correction_effect.json","w"),indent=2)
print("[完成] plot_correction_effect.json")
