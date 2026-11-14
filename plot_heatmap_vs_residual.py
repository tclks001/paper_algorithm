"""
并排对照：
  左 = 用户 hot_radius.png 口径：小三角形外接圆半径 phi(t)，画在【三角形重心】
  右 = 思路 C：顶点 1-ring 平衡残差 r_i = max-min(phi over 1-ring)，画在【顶点】
取 face 0 正三角面展开，英文标注避免字体缺失。
"""
import numpy as np, math
from scipy.spatial import ConvexHull
import improve_corrected as ic
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

n=27
P0,fi,I,J,isedge=ic.gen_meta(n)
A,B,C=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,C-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1, Y@e2]

h=ConvexHull(P0); tris=h.simplices
phi=np.array([math.degrees(ic.sph_circumradius(P0[t[0]],P0[t[1]],P0[t[2]])) for t in tris])
ring=[[] for _ in range(len(P0))]
for t,r in zip(tris,phi):
    for k in t: ring[k].append(r)
res=np.array([max(a)-min(a) for a in ring])

# 只保留 face 0 内的三角形与顶点
fc=normalize(A+B+C)
def in_face0(p, tol=1e-6):
    return np.dot(p,A)>np.dot(fc,A)-1e-2   # 粗筛：仍在 face0 半球一侧
cent=np.array([normalize(P0[t].sum(axis=0)) for t in tris])
keep=np.array([np.dot(cent[k],fc)>0.985 for k in range(len(tris))])
sel_v=np.where(fi==0)[0]
P2=to2d(P0)
tri2=np.array([*to2d(np.array([A,B,C])), to2d(A)[0]])

fig,ax=plt.subplots(1,2,figsize=(12,5.2))
cc=np.array([to2d(normalize(P0[t].sum(axis=0)))[0] for t in tris[keep]])
sc0=ax[0].scatter(cc[:,0],cc[:,1],c=phi[keep],s=9,cmap='coolwarm',vmin=phi.min(),vmax=phi.max())
sc1=ax[1].scatter(P2[sel_v,0],P2[sel_v,1],c=res[sel_v],s=9,cmap='viridis',
                  vmin=0,vmax=np.percentile(res[sel_v],99))
for a in ax:
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal')
    a.set_xlabel('x'); a.set_ylabel('y'); a.axis('off')
ax[0].set_title('Heat map (user hot_radius.png)\nphi(t) on each small TRIANGLE\nred = largest circumradius (sparsest)')
ax[1].set_title('Idea C: 1-ring balance residual\nr_i on each VERTEX\n= local non-uniformity (movability)')
plt.colorbar(sc0,ax=ax[0],shrink=.72,label='phi(t), deg')
plt.colorbar(sc1,ax=ax[1],shrink=.72,label='r_i, deg')
fig.tight_layout()
fig.savefig("E:/毕业论文/算法/fig_heatmap_vs_residual.png",dpi=140)
print("[出图] fig_heatmap_vs_residual.png")
print(f"  phi 范围 [{phi.min():.6f},{phi.max():.6f}]deg   r_i 中位={np.median(res):.6f}deg")
