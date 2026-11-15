"""
联合优化版（rho=1.488060）的可视化：热力图 / 残差图 / 位移箭头图
================================================================
口径与 plot_correction_effect.py / plot_final_symmetric.py 完全一致：
  - face 0，406 个格点，按 (I,J) 建索引；构造网格三角形 up/down 共 n^2=729 个
  - 热力图 phi(t)=球面外接圆半径，画在三角形重心
  - 残差 r_i = max-min(phi over 1-ring)，画在初始格点位置
  - 修正前后用同一组三角形索引与同一 colorbar 范围

箭头长度修正（本次重点）：
  旧图 AMP=180 使最长箭头约 0.42 数据单位，而格距仅约 0.039 => 跨约 10 格，严重重叠。
  本脚本把两版共用 AMP 定为「最长箭头 = 0.42 * 格距」，并额外提供
  「等长箭头 + 颜色编码幅度」版本，彻底消除重叠。

输出：
  fig_joint_2x2.png              P0 / 联合版 热力图 + 残差图 (2x2)
  fig_joint_arrows_compare.png   方案C(优化前) vs 联合版(优化后) 短箭头并排
  fig_joint_arrows_after.png     联合版单独大图：等长箭头 + 幅度颜色
  fig_joint_residual_3way.png    P0 / 方案C / 联合版 残差三方对照
  plot_joint_effect.json         定量统计
"""
import numpy as np, math, json, csv
from scipy.spatial import ConvexHull
import improve_corrected as ic
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="E:/毕业论文/算法"; n=27
def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

def load(path):
    rows=list(csv.DictReader(open(path)))
    P0=np.array([[float(r['x']),float(r['y']),float(r['z'])] for r in rows])
    P =np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in rows])
    fi=np.array([int(r['face_idx']) for r in rows])
    I =np.array([int(r['i']) for r in rows]); J=np.array([int(r['j']) for r in rows])
    return P0,P,fi,I,J

P0,PJ,fi,I,J=load(OUT+"/theta_m0_joint_points_n27.csv")     # 联合优化后
P0C,PC,fiC,IC,JC=load(OUT+"/symmetric_points_final_C.csv")  # 优化前 = 方案 C

# ---------- 正确性断言 ----------
assert len(P0)==10*n*n+2==7292 and len(P0C)==7292
assert np.abs(P0-P0C).max()<1e-12, "两份 CSV 的 P0 顺序不一致"
assert np.array_equal(fi,fiC) and np.array_equal(I,IC) and np.array_equal(J,JC), "格点元数据不一致"

# ---------- face 0 局部索引 / 2D 投影 ----------
sel=np.where(fi==0)[0]
loc={(int(I[k]),int(J[k])):h for h,k in enumerate(sel)}
A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tri2=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])
edge_len=float(np.linalg.norm(to2d(B)-to2d(A)))
grid=edge_len/n
print(f"face0 格点数={len(sel)}  三角形边长(2D)={edge_len:.6f}  格距={grid:.6f}")

tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
assert len(tris)==n*n==729
P2=to2d(P0[sel])

def analyze(P):
    Pf=P[sel]
    phi=np.array([math.degrees(ic.sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    cent=np.array([normalize(Pf[t].sum(axis=0)) for t in tris])
    ring=[[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    res=np.array([max(a)-min(a) for a in ring])
    return phi,to2d(cent),res

def rho_global(P):
    h=ConvexHull(P); t=h.simplices
    a0,b0,c0=P[t[:,0]],P[t[:,1]],P[t[:,2]]
    aa=np.arccos(np.clip(np.sum(a0*b0,1),-1,1))
    bb=np.arccos(np.clip(np.sum(b0*c0,1),-1,1))
    cc=np.arccos(np.clip(np.sum(c0*a0,1),-1,1))
    s=(aa+bb+cc)/2
    val=np.sin(s-aa)*np.sin(s-bb)*np.sin(s-cc)/np.sin(s)
    return math.degrees(float((2*np.arctan(np.sqrt(np.clip(val,0,None)))).max()))

def st(x):
    return dict(min=float(x.min()),max=float(x.max()),range=float(x.max()-x.min()),
                mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)))

phi0,cc0,res0=analyze(P0)
phiC,ccC,resC=analyze(PC)
phiJ,ccJ,resJ=analyze(PJ)
r0,rC,rJ=rho_global(P0),rho_global(PC),rho_global(PJ)
print(f"全局 rho_max: P0={r0:.6f}  方案C={rC:.6f}  联合版={rJ:.6f}")
assert abs(rJ-1.488060)<1e-4 and abs(rC-1.496089)<1e-4, "点集与已记录 rho 不符"

s0,sC,sJ=st(phi0),st(phiC),st(phiJ)
t0,tC,tJ=st(res0),st(resC),st(resJ)
print("\n=== 热力图 phi(t) ===")
for tag,s in [('P0',s0),('scheme C',sC),('JOINT',sJ)]:
    print(f"  {tag:10s} range={s['range']:.6f} std={s['std']:.6f} min={s['min']:.6f} max={s['max']:.6f}")
print("=== 1-ring 残差 ===")
for tag,s in [('P0',t0),('scheme C',tC),('JOINT',tJ)]:
    print(f"  {tag:10s} median={s['median']:.6f} max={s['max']:.6f} mean={s['mean']:.6f}")

# ---------- 箭头数据 ----------
dC=to2d(PC[sel])-P2
dJ=to2d(PJ[sel])-P2
mC=np.linalg.norm(dC,axis=1); mJ=np.linalg.norm(dJ,axis=1)
def ang3(P):
    return np.degrees(np.arccos(np.clip(np.sum(P0[sel]*P[sel],axis=1),-1,1)))
aC,aJ=ang3(PC),ang3(PJ)
LMAX=0.42*grid
AMP=LMAX/max(mC.max(),mJ.max())
print(f"\n旧图 AMP=180 -> 最长箭头 {180*max(mC.max(),mJ.max()):.4f} = "
      f"{180*max(mC.max(),mJ.max())/grid:.1f} 格距 (严重重叠)")
print(f"新图 AMP={AMP:.2f} -> 最长箭头 {AMP*max(mC.max(),mJ.max()):.5f} = "
      f"{AMP*max(mC.max(),mJ.max())/grid:.2f} 格距 (无跨格重叠)")
print(f"  位移(3D角距) 方案C mean={aC.mean():.6f} max={aC.max():.6f} | "
      f"联合版 mean={aJ.mean():.6f} max={aJ.max():.6f} deg")

mid=[normalize(B+Cc),normalize(A+Cc),normalize(A+B)]
def base_panel(ax):
    ax.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2)
    for vv,mm in zip([A,B,Cc],mid):
        ln=to2d(np.array([vv,mm])); ax.plot(ln[:,0],ln[:,1],'k--',lw=.5,alpha=.28)
    ax.scatter(P2[:,0],P2[:,1],s=3.5,c='0.80',edgecolors='none',zorder=2)
    ax.set_aspect('equal'); ax.axis('off')

KW=dict(angles='xy',scale_units='xy',scale=1.0,width=.0016,
        headwidth=3,headlength=4,alpha=.88,zorder=4)

# ---------- 图1: 2x2 热力图 / 残差图 ----------
fig,ax=plt.subplots(2,2,figsize=(11.5,10))
pvmin=min(s0['min'],sJ['min']); pvmax=max(s0['max'],sJ['max'])
rvmax=max(t0['max'],tJ['max'])
for a in ax.ravel(): base_panel(a)
for j,(cc,phi,s,ttl) in enumerate([(cc0,phi0,s0,'BEFORE (P0 baseline)'),
                                   (ccJ,phiJ,sJ,'AFTER (joint optimization)')]):
    sc=ax[0,j].scatter(cc[:,0],cc[:,1],c=phi,s=11,cmap='coolwarm',vmin=pvmin,vmax=pvmax,zorder=3)
    ax[0,j].set_title(f'Heat map phi(t) - {ttl}\nrange={s["range"]:.6f}  std={s["std"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[0,j],shrink=.8,label='phi(t), deg')
for j,(res,s,ttl) in enumerate([(res0,t0,'BEFORE (P0 baseline)'),
                                (resJ,tJ,'AFTER (joint optimization)')]):
    sc=ax[1,j].scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rvmax,zorder=3)
    ax[1,j].set_title(f'1-ring residual r_i - {ttl}\nmedian={s["median"]:.6f}  max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=ax[1,j],shrink=.8,label='r_i, deg')
fig.suptitle(f'face 0, n=27 | BEFORE vs AFTER joint (m0,a,theta) optimization\n'
             f'global rho_max {r0:.6f} -> {rJ:.6f} deg  |  phi std {s0["std"]:.6f} -> {sJ["std"]:.6f}',fontsize=11)
fig.tight_layout(); fig.savefig(OUT+"/fig_joint_2x2.png",dpi=150)
print("\n[出图] fig_joint_2x2.png")

# ---------- 图2: 短箭头并排（方案C vs 联合版） ----------
fig2,ax2=plt.subplots(1,2,figsize=(13.5,6.4))
for a,(d,mag,title) in zip(ax2,[
    (dC,aC,f'BEFORE: scheme C  (rho={rC:.6f} deg)\nm(dev)~a*dev,  |disp| mean={aC.mean():.4f} deg'),
    (dJ,aJ,f'AFTER: joint optimization  (rho={rJ:.6f} deg)\nm(dev)~m0,  |disp| mean={aJ.mean():.4f} deg')]):
    base_panel(a)
    a.quiver(P2[:,0],P2[:,1],d[:,0]*AMP,d[:,1]*AMP,color='tab:red',**KW)
    a.set_title(title,fontsize=9.5,pad=8)
fig2.suptitle(f'Displacement arrows P0 -> optimized | SHORT: x{AMP:.1f}, '
              f'longest = 0.42 grid spacing | grey dots = initial P0 | both panels share ONE scale',
              fontsize=10.5,y=0.985)
fig2.tight_layout(rect=[0,0,1,0.86]); fig2.savefig(OUT+"/fig_joint_arrows_compare.png",dpi=150)
print("[出图] fig_joint_arrows_compare.png")

# ---------- 图3: 联合版单独大图（等长箭头 + 幅度颜色，完全无重叠） ----------
fig3,ax3=plt.subplots(figsize=(8.6,7.6))
base_panel(ax3)
nz=mJ>1e-12
U=np.zeros_like(dJ)
U[nz]=dJ[nz]/mJ[nz,None]
U*=0.36*grid
sc=ax3.quiver(P2[:,0],P2[:,1],U[:,0],U[:,1],aJ,cmap='viridis',
              angles='xy',scale_units='xy',scale=1.0,width=.0018,
              headwidth=3.5,headlength=4.5,alpha=.92,zorder=4)
plt.colorbar(sc,ax=ax3,shrink=.78,label='|disp| (deg)')
ax3.set_title(f'AFTER joint optimization - UNIT-LENGTH direction arrows\n'
              f'all arrows = {0.36:.2f} grid spacing, colour = true displacement magnitude\n'
              f'rho={rJ:.6f} deg  |  zero disp = {int((aJ<1e-9).sum())} points',fontsize=10)
fig3.tight_layout(); fig3.savefig(OUT+"/fig_joint_arrows_after.png",dpi=150)
print("[出图] fig_joint_arrows_after.png")

# ---------- 图4: 三方残差 ----------
fig4,ax4=plt.subplots(1,3,figsize=(15,4.9))
rv=max(t0['max'],tC['max'],tJ['max'])
for a,(res,s,ttl) in zip(ax4,[(res0,t0,'P0 baseline'),(resC,tC,'scheme C (before joint)'),
                              (resJ,tJ,'joint optimized (after)')]):
    base_panel(a)
    sc=a.scatter(P2[:,0],P2[:,1],c=res,s=13,cmap='viridis',vmin=0,vmax=rv,zorder=3)
    a.set_title(f'{ttl}\nmedian={s["median"]:.6f}  max={s["max"]:.6f} deg',fontsize=10)
    plt.colorbar(sc,ax=a,shrink=.8,label='r_i, deg')
fig4.suptitle('1-ring residual: joint optimization improves rho_max/phi-std but worsens local balance',fontsize=11)
fig4.tight_layout(); fig4.savefig(OUT+"/fig_joint_residual_3way.png",dpi=150)
print("[出图] fig_joint_residual_3way.png")

json.dump(dict(phi_before=s0,phi_schemeC=sC,phi_joint=sJ,
               res_before=t0,res_schemeC=tC,res_joint=tJ,
               rho_global=dict(P0=r0,schemeC=rC,joint=rJ),
               arrows=dict(old_AMP=180.0,new_AMP=float(AMP),grid_spacing=float(grid),
                           edge_len_2d=edge_len,longest_arrow=float(LMAX),
                           longest_in_grid=0.42,unit_arrow_in_grid=0.36,
                           old_longest_in_grid=float(180*max(mC.max(),mJ.max())/grid),
                           disp_mean_schemeC=float(aC.mean()),disp_max_schemeC=float(aC.max()),
                           disp_mean_joint=float(aJ.mean()),disp_max_joint=float(aJ.max()),
                           zero_disp_joint=int((aJ<1e-9).sum())),
               points=int(len(sel)),tris=int(len(tris)),n=n),
          open(OUT+"/plot_joint_effect.json","w"),indent=2)
print("[完成] plot_joint_effect.json")
