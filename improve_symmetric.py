"""
【正式版】对称边界感知修正算法（取代 improve_corrected.py 的不协变实现）
=========================================================================
设计（对应 12:15 用户拼图 = 1/6 基本域 + 对称复制的严格实现）：
  1) 以 gen_meta 生成 n=27 的球面质心构造 P0（|P0|=10n^2+2=7292）。
  2) 定义"初步位移场" d0(p)  = m(dev(p)) * u(p)，其中
     m(dev) = m0 + a*dev + c*dev^2（平方参数化保证单调非负），
     u 为局部方向规则（候选 A 用原混合规则；候选 B 只用朝面心径向）。
  3) 用二十面体全群 G120 做 Reynolds 投影（= 群轨道复制的等价形式）：
     v(p) = (1/|G|) * sum_g  g^{-1} * d0(g p)
     这是到 G 不变子空间的正交投影，必然严格满足 v(g p) = g v(p)。
  4) 幅度系数整体重标定：把 s 吸收进 (m0,a,c)，不保留额外 scale 参数。
  5) p' = ExpMap(p, |v(p)|, v/|v|)；12 个顶点固定不动。

正确性门禁（本脚本全部自检）：
  - 点数 7292
  - 严格多重集对称：120 个群元逐一要求双射 + 距离 < 1e-9
  - rho_max <= 1.5 deg
  - 镜像协变角偏差 median ~ 0
  - 三条棱上沿棱切向分量严格为 0（无环流）
交付：symmetric_points_n27.csv / symmetric_coef_n27.json / v_symmetric_final.npy
"""
import numpy as np, math, json, csv, os
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
m=corr.m_of(raw)                       # m0+a*dev+c*dev^2
u_mix=np.array([corr.dir_of(k) for k in range(len(P0))])   # 候选A：原混合规则
u_rad=corr.u_rad.copy()                                     # 候选B：纯径向
N=len(P0)
print(f"N={N} (应为 {10*n*n+2})  顶点数={is_vertex.sum()}")

# ---------- 群 ----------
V=ic.V
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
print(f"群元数={len(G120)} (60 proper + 60 improper)")

tree=cKDTree(P0)
IDX=[tree.query((g@P0.T).T)[1] for g in G120]

def reynolds(d0):
    """d0: (N,3) 位移矢量场；返回 G 不变投影"""
    acc=np.zeros((N,3))
    for g,j in zip(G120,IDX):
        acc += (g.T @ d0[j].T).T
    return acc/len(G120)

def build_P(v,scale=1.0):
    P=np.empty_like(P0); mag=np.linalg.norm(v,axis=1)
    for k in range(N):
        if is_vertex[k] or mag[k]<1e-12: P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],scale*mag[k],normalize(v[k]))
    return P

def rho(P):
    h=ConvexHull(P)
    return math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))

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
    q=cKDTree(base).query((S@base.T).T)[1]
    def td(Z):
        d=Z[sel]-base; return d-np.sum(d*base,axis=1)[:,None]*base
    d=td(P); Sd=(S@d.T).T
    num=np.sum(Sd*d[q],axis=1)
    den=np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[q],axis=1)+1e-30
    return float(np.median(np.degrees(np.arccos(np.clip(num/den,-1,1)))))

def edge_circulation(P):
    """三条棱上位移沿棱分量绝对值之和（对称场应为 0）"""
    sel=np.where(fi==0)[0]; base=P0[sel]
    d=P[sel]-base; d=d-np.sum(d*base,axis=1)[:,None]*base
    A,B,Cc=V[0],V[4],V[8]
    out=[]
    for cond,u in [(lambda i,j:j==0,normalize(B-A)),
                   (lambda i,j:j==i,normalize(Cc-A)),
                   (lambda i,j:i==n,normalize(Cc-B))]:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        out.append(float(np.mean([abs(np.dot(d[h],u)) for h in ks])))
    return out

# ---------- 候选 A / B ----------
cands={
 "A_mixed_reynolds": reynolds(m[:,None]*u_mix),
 "B_radial_reynolds": reynolds(m[:,None]*u_rad),
}
results={}
for name,v in cands.items():
    print(f"\n=== 候选 {name} ===")
    print(f"  |v| mean={np.linalg.norm(v,axis=1).mean():.6f} max={np.linalg.norm(v,axis=1).max():.6f} deg")
    best=None
    for s in [0.8,1.0,1.2,1.4,1.5,1.6,1.8,2.0,2.2,2.5]:
        r=rho(build_P(v,s))
        print(f"    s={s:.2f}  rho_max={r:.6f}deg")
        if best is None or r<best[1]: best=(s,r)
    P=build_P(v,best[0]); bad,worst=strict_sym(P)
    print(f"  -> 最优 s={best[0]:.2f} rho={best[1]:.6f}  "
          f"严格对称失败={bad}/120 最大偏差={worst:.9f}deg")
    print(f"     镜像角偏差 median={mirror_median(P):.4f}deg  "
          f"三棱沿棱分量={np.round(edge_circulation(P),9)}")
    results[name]=dict(scale=best[0],rho=best[1],sym_bad=bad,sym_worst=worst,
                       mirror=mirror_median(P),edge=edge_circulation(P))

best_name=min(results,key=lambda k:results[k]["rho"])
print(f"\n=== 选用 {best_name} (rho 最优) ===")
v=cands[best_name]; s=results[best_name]["scale"]
P=build_P(v,s)

# ---------- 把 s 吸收进系数 ----------
m0,a,c=raw[0]**2,raw[1]**2,raw[2]**2
coef=dict(m0=s*m0,a=s*a,c=s*c,scale_absorbed=s,source=best_name,
          m_formula="m(dev)=m0+a*dev+c*dev^2",
          note="Reynolds G120 projection of the base field; 12 vertices fixed; "
               "axis points automatically have zero tangential component")
print(f"  最终系数: m0={coef['m0']:.8f}  a={coef['a']:.8f}  c={coef['c']:.3e}  (已吸收 s={s})")

# ---------- 门禁 ----------
bad,worst=strict_sym(P); r=rho(P)
assert len(P)==7292
assert bad==0, f"对称校验失败 {bad}/120"
assert r<=1.5, f"rho_max={r} 超标"
mirror=mirror_median(P); edges=edge_circulation(P)
assert mirror<1e-3, f"镜像协变偏差 {mirror}"
assert max(edges)<1e-9, f"存在沿棱环流 {edges}"
print(f"\n[门禁全部通过] 点数={len(P)}  rho_max={r:.6f}deg  对称失败={bad}/120  "
      f"偏差={worst:.9f}deg  镜像median={mirror:.6f}deg  三棱沿棱={np.round(edges,12)}")

# ---------- 交付 ----------
np.save(OUT+"/v_symmetric_final.npy",v)
json.dump({**coef,"rho_max":r,"strict_sym_bad":bad,"mirror_median_deg":mirror,
           "edge_circulation":edges,"candidates":results},
          open(OUT+"/symmetric_coef_n27.json","w"),indent=2)
with open(OUT+"/symmetric_points_n27.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j",
                                 "dev_deg","disp_deg","is_edge","is_vertex"])
    for k in range(N):
        d=math.degrees(math.acos(np.clip(np.dot(P0[k],P[k]),-1,1)))
        w.writerow([P0[k,0],P0[k,1],P0[k,2],P[k,0],P[k,1],P[k,2],
                    int(fi[k]),int(I[k]),int(J[k]),dev[k],d,int(isedge[k]),int(is_vertex[k])])
print("[交付] symmetric_points_n27.csv / symmetric_coef_n27.json / v_symmetric_final.npy")
