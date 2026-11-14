"""
补充对照：保留"边界感知切向"的严格对称修正
=============================================
问题：Reynolds 投影原场后，切向（原版是统一环绕、非协变）被完全抵消，
      得到纯径向场。用户 12:15 拼图在边缘保留切向偏转，故构造一个
      **协变切向**版本：沿棱方向乘以关于棱中点的奇函数 sign(s)，
      s = angle(p,X)-angle(p,Y)，使 2 重旋转下切向自动反号 => 协变。
      再做 G120 Reynolds 投影，得到"保留切向且严格对称"的场。

检验：rho_max、严格对称、镜像偏差、三条棱沿棱分量（对称场应为反号而非统一同号）。
"""
import numpy as np, math, json
from scipy.spatial import cKDTree, ConvexHull
import improve_corrected as ic

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
def rot_axis(k,theta):
    k=normalize(k); c=math.cos(theta); s=math.sin(theta)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

OUT="E:/毕业论文/算法"
n=27
P0,fi,I,J,isedge=ic.gen_meta(n)
C,dev,is_vertex,u_edge,min_bary,Cs=ic.build_fields(P0,fi)
corr=ic.Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
raw=np.array(json.load(open(OUT+"/corrected_coef_n27.json"))["raw"],float)
m=corr.m_of(raw)
N=len(P0); band=0.08
wb=np.clip((band-min_bary)/band,0.0,1.0)

# ---------- 协变切向场 d0 ----------
V=ic.V; faces=ic.faces
u_rad=corr.u_rad
u_edge_cov=np.zeros((N,3))
for k in range(N):
    f=faces[fi[k]]; AA,BB,CC=V[f[0]],V[f[1]],V[f[2]]
    edges=[(AA,BB),(BB,CC),(CC,AA)]
    best=None; be=1e9
    for (X,Y) in edges:
        e=abs(ic.angle(P0[k],X)+ic.angle(P0[k],Y)-ic.angle(X,Y))
        if e<be: be=e; best=(X,Y)
    X,Y=best
    ne=normalize(np.cross(X,Y))
    ue=normalize(np.cross(P0[k],ne))
    s=ic.angle(P0[k],X)-ic.angle(P0[k],Y)     # 关于棱中点的奇函数
    u_edge_cov[k]=ue*(1.0 if s>=0 else -1.0)
d0=np.empty((N,3))
for k in range(N):
    u=normalize((1-wb[k])*u_rad[k]+wb[k]*u_edge_cov[k])
    d0[k]=m[k]*u
d0[is_vertex]=0.0
print(f"协变切向场 |d0| mean={np.linalg.norm(d0,axis=1).mean():.6f} "
      f"max={np.linalg.norm(d0,axis=1).max():.6f} deg")

# ---------- 群 ----------
av=normalize(V[1]); af=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(av,2*math.pi/5),rot_axis(av,-2*math.pi/5),
      rot_axis(af,2*math.pi/3),rot_axis(af,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); k=tuple(np.round(M.ravel(),8))
    if k in seen: continue
    seen.add(k); G60.append(M)
    for g in gens: todo.append(g@M)
G120=G60+[-g for g in G60]
tree=cKDTree(P0); IDX=[tree.query((g@P0.T).T)[1] for g in G120]
acc=np.zeros((N,3))
for g,j in zip(G120,IDX): acc+=(g.T@d0[j].T).T
v=acc/len(G120)
print(f"Reynolds 后 |v| mean={np.linalg.norm(v,axis=1).mean():.6f} "
      f"max={np.linalg.norm(v,axis=1).max():.6f} deg")

def build_P(scale):
    P=np.empty_like(P0); mag=np.linalg.norm(v,axis=1)
    for k in range(N):
        if is_vertex[k] or mag[k]<1e-12: P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],scale*mag[k],normalize(v[k]))
    return P
def rho(P):
    h=ConvexHull(P)
    return math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))
def strict(P):
    t=cKDTree(P); bad=0
    for g in G120:
        d,j=t.query((g@P.T).T)
        if len(np.unique(j))!=len(P) or d.max()>1e-9: bad+=1
    return bad
def edge_along(P):
    """三条棱上 d·u 的【均值】（统一环绕 => 非零；对称反号 => ~0）与绝对值均值"""
    sel=np.where(fi==0)[0]; base=P0[sel]
    d=P[sel]-base; d=d-np.sum(d*base,axis=1)[:,None]*base
    A,B,Cc=V[0],V[4],V[8]
    out=[]
    for cond,u in [(lambda i,j:j==0,normalize(B-A)),
                   (lambda i,j:j==i,normalize(Cc-A)),
                   (lambda i,j:i==n,normalize(Cc-B))]:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        pr=np.array([np.dot(d[h],u) for h in ks])
        out.append((float(pr.mean()),float(np.abs(pr).mean())))
    return out

best=None
print("\n=== scale 扫描 ===")
for s in [0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
    r=rho(build_P(s)); print(f"  s={s:.2f}  rho_max={r:.6f}deg")
    if best is None or r<best[1]: best=(s,r)
P=build_P(best[0])
print(f"\n  -> 最优 s={best[0]:.2f} rho={best[1]:.6f}  严格对称失败={strict(P)}/120")
print(f"     三条棱 (符号均值, 绝对值均值) = {np.round(edge_along(P),9)}")
print(f"     对照：纯径向对称版 rho=1.492573, 三棱均为 (0,0)")
np.save(OUT+"/v_covariant_tangent.npy",v)

# ---------- 正式交付与门禁 ----------
bad=strict(P)
sym_mean=[abs(x[0]) for x in edge_along(P)]
assert len(P)==7292
assert bad==0, f"对称校验失败 {bad}/120"
assert best[1]<=1.5, f"rho_max={best[1]} 超标"
assert max(sym_mean)<1e-6, f"三条棱仍存在统一环绕 {sym_mean}"
# 镜像协变
G=normalize(V[0]+V[4]+V[8]); nA=normalize(np.cross(G,V[0]))
S=np.eye(3)-2*np.outer(nA,nA); sel=np.where(fi==0)[0]; base=P0[sel]
q=cKDTree(base).query((S@base.T).T)[1]
def td(Z):
    d=Z[sel]-base; return d-np.sum(d*base,axis=1)[:,None]*base
d=td(P); Sd=(S@d.T).T
num=np.sum(Sd*d[q],axis=1); den=np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[q],axis=1)+1e-30
mirror=float(np.median(np.degrees(np.arccos(np.clip(num/den,-1,1)))))
assert mirror<1e-3, f"镜像协变偏差 {mirror}"
print(f"\n[门禁全部通过] 点数={len(P)} rho_max={best[1]:.6f} 对称失败={bad}/120 "
      f"镜像median={mirror:.6f} 三棱符号均值={np.round(sym_mean,12)}")

coef=dict(m0=raw[0]**2*best[0],a=raw[1]**2*best[0],c=raw[2]**2*best[0],
          scale_absorbed=best[0],variant="C_covariant_tangent",
          construction="edge tangent multiplied by odd function sign(angle(p,X)-angle(p,Y)) "
                       "about edge midpoint, then G120 Reynolds projection",
          rho_max=best[1],strict_sym_bad=bad,mirror_median_deg=mirror,
          edge_signed_mean=sym_mean,edge_abs_mean=[x[1] for x in edge_along(P)],
          note="This is the CORRECT boundary-aware algorithm: strictly 120-fold symmetric, "
               "edge points still glide along edges with sign flipping at midpoints, no circulation.")
json.dump(coef,open(OUT+"/symmetric_coef_final_C.json","w"),indent=2)
import csv as _csv
with open(OUT+"/symmetric_points_final_C.csv","w",newline="") as f:
    w=_csv.writer(f); w.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j",
                                  "dev_deg","disp_deg","is_edge","is_vertex"])
    for k in range(N):
        dd=math.degrees(math.acos(np.clip(np.dot(P0[k],P[k]),-1,1)))
        w.writerow([P0[k,0],P0[k,1],P0[k,2],P[k,0],P[k,1],P[k,2],
                    int(fi[k]),int(I[k]),int(J[k]),dev[k],dd,int(isedge[k]),int(is_vertex[k])])
json.dump(dict(best_scale=best[0],rho=best[1],strict_sym_bad=int(bad),
               edge_along=edge_along(P),mirror_median_deg=mirror),
          open(OUT+"/compare_covariant_tangent.json","w"),indent=2)
print("[交付] symmetric_points_final_C.csv / symmetric_coef_final_C.json / v_covariant_tangent.npy")
print("[完成] compare_covariant_tangent.json")
