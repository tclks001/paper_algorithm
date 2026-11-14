"""
诊断 fig_correction_displacement_face0.png 中"靠近棱的点呈旋转"的现象
=====================================================================
三个检验：
 1) 镜像协变性：face 0 内关于中线(过顶点 A 与面心)的反射 S，
    检查 d(S p) 是否 = S d(p)。对称场应 cos≈1。
 2) 三条棱的切向符号：若 (X,Y) 取自固定有序边表 [(A,B),(B,C),(C,A)]，
    三条棱的 u_edge 会形成统一绕行 A->B->C->A，视觉上即"旋转"。
    检验每条棱上点的 d·u_edge 符号，以及棱中点两侧是否反号。
 3) 2D 投影畸变：比较球面切向位移与正交投影后的方向夹角，排除视觉假象。
并出对比箭头图：原版代码场 vs Reynolds 严格对称场。
"""
import numpy as np, math, json
from scipy.spatial import cKDTree
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
v=np.load("E:/毕业论文/算法/v120_symmetric.npy")
Psym=np.empty_like(P0)
for k in range(len(P0)):
    mag=np.linalg.norm(v[k])
    if is_vertex[k] or mag<1e-12: Psym[k]=P0[k]; continue
    Psym[k]=ic.exp_map(P0[k],1.5*mag,normalize(v[k]))

sel=np.where(fi==0)[0]
P0f=P0[sel]
A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tri2=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])

def tangential_disp(P):
    d=P[sel]-P0f
    d=d-np.sum(d*P0f,axis=1)[:,None]*P0f     # 去掉径向，留切平面分量
    return d

# ---------- 1) 镜像协变性 ----------
G=normalize(A+B+Cc)
nA=normalize(np.cross(G,A))
S=np.eye(3)-2*np.outer(nA,nA)
print(f"反射 S 校验: |S@B - C| = {np.linalg.norm(S@B-Cc):.3e} (应 ~0)  det(S)={np.linalg.det(S):.1f}")
tree=cKDTree(P0f)
_,q=tree.query((S@P0f.T).T)
def mirror_err(P):
    d=tangential_disp(P)
    Sd=(S@d.T).T
    num=np.sum(Sd*d[q],axis=1)
    den=np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[q],axis=1)+1e-30
    cos=np.clip(num/den,-1,1)
    return np.degrees(np.arccos(cos)),cos
for tag,P in [("原版边界感知修正",Popt),("Reynolds 对称化场",Psym)]:
    ang,cos=mirror_err(P)
    print(f"  {tag:<20}: 镜像角偏差 median={np.median(ang):7.2f}deg  max={ang.max():7.2f}deg  "
          f"cos<0 的点数={(cos<0).sum()}/{len(cos)}")

# ---------- 2) 三条棱的切向符号 ----------
# face 0 的 (I,J): 边 j=0 是 AB, 边 j=i 是 AC, 边 i=n 是 BC
edge_defs=[("AB (j=0)",lambda i,j:j==0,normalize(B-A)),
           ("AC (j=i)",lambda i,j:j==i,normalize(Cc-A)),
           ("BC (i=n)",lambda i,j:i==n,normalize(Cc-B))]
print("\n=== 三条棱上位移沿棱方向的符号（u 按 A->B->C->A 绕行定向） ===")
for tag,P in [("原版",Popt),("对称化",Psym)]:
    d=tangential_disp(P)
    print(f"  [{tag}]")
    for name,cond,u in edge_defs:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        proj=np.array([np.dot(d[h],u) for h in ks])
        pos=int((proj>1e-9).sum()); neg=int((proj<-1e-9).sum())
        print(f"    {name:<12}: 点数={len(ks):3d}  正号={pos:3d} 负号={neg:3d} "
              f"均值={proj.mean():+.6f}  -> {'统一绕行(旋转)' if abs(pos-neg)>len(ks)*0.7 else '两侧反号(对称)'}")

# ---------- 3) 2D 投影畸变 ----------
d3=tangential_disp(Popt)
d2=np.c_[d3@e1,d3@e2]
d2_of_points=to2d(Popt[sel])-to2d(P0f)
dev2=np.degrees(np.arccos(np.clip(np.sum(d2*d2_of_points,axis=1)/
    (np.linalg.norm(d2,axis=1)*np.linalg.norm(d2_of_points,axis=1)+1e-30),-1,1)))
print(f"\n=== 2D 投影畸变（切向量投影 vs 点的投影差）===")
print(f"  角度偏差 median={np.median(dev2):.4f}deg  max={dev2.max():.4f}deg  -> 投影不是'旋转'的成因")

# ---------- 对比箭头图 ----------
AMP=180.0
fig,ax=plt.subplots(1,2,figsize=(13,6))
for a,(tag,P) in zip(ax,[("ORIGINAL code field (paper 3.3.4)",Popt),("Reynolds STRICTLY symmetric field",Psym)]):
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.3)
    dd=to2d(P[sel])-to2d(P0f)
    a.quiver(to2d(P0f)[:,0],to2d(P0f)[:,1],dd[:,0]*AMP,dd[:,1]*AMP,
             angles='xy',scale_units='xy',scale=1.0,color='tab:red',width=.0022,alpha=.8)
    a.set_aspect('equal'); a.axis('off'); a.set_title(f'{tag}\narrows x{AMP:.0f}',fontsize=10)
fig.suptitle('Why the edge arrows look like a rotation: fixed ordered edge list vs covariant field',fontsize=11)
fig.tight_layout(); fig.savefig("E:/毕业论文/算法/fig_rotation_vs_symmetric.png",dpi=145)
print("\n[出图] fig_rotation_vs_symmetric.png")

json.dump({"mirror_median_original":float(np.median(mirror_err(Popt)[0])),
           "mirror_median_symmetric":float(np.median(mirror_err(Psym)[0])),
           "mirror_max_original":float(mirror_err(Popt)[0].max()),
           "mirror_max_symmetric":float(mirror_err(Psym)[0].max()),
           "proj_distortion_median":float(np.median(dev2)),
           "proj_distortion_max":float(dev2.max())},
          open("E:/毕业论文/算法/diagnose_rotation.json","w"),indent=2)
print("[完成] diagnose_rotation.json")
