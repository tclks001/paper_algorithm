"""
三方对照：P0 / 原版边界感知修正 / Reynolds 对称化修正(s=1.5)
==============================================================
用户两组实验的补充诊断：修正后残差 max 暴涨 6 倍，需定位高残差点并判断
是"原版实现缺陷"还是"修正本身固有代价"。
口径与 plot_correction_effect.py 完全一致（face 0、同一组构造三角形）。
"""
import numpy as np, math, json
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

v=np.load("E:/毕业论文/算法/v120_symmetric.npy")
Psym=np.empty_like(P0)
for k in range(len(P0)):
    mag=np.linalg.norm(v[k])
    if is_vertex[k] or mag<1e-12: Psym[k]=P0[k]; continue
    Psym[k]=ic.exp_map(P0[k],1.5*mag,normalize(v[k]))

sel=np.where(fi==0)[0]
loc={(int(I[k]),int(J[k])):i for i,k in enumerate(sel)}
A,B,Cc=ic.V[0],ic.V[4],ic.V[8]
e1=normalize(B-A); nn=normalize(np.cross(B-A,Cc-A)); e2=normalize(np.cross(nn,e1))
def to2d(X):
    Y=np.atleast_2d(X)-A; return np.c_[Y@e1,Y@e2]
tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
P2=to2d(P0[sel]); tri2=np.array([*to2d(np.array([A,B,Cc])),to2d(A)[0]])

def analyze(P):
    Pf=P[sel]
    phi=np.array([math.degrees(ic.sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    cent=np.array([normalize(Pf[t].sum(axis=0)) for t in tris])
    ring=[[] for _ in range(len(sel))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    return phi,to2d(cent),np.array([max(a)-min(a) for a in ring])

sets=[("P0 baseline",P0),("boundary-aware (paper 3.3.4)",Popt),("Reynolds symmetric s=1.5",Psym)]
R={}
for tag,P in sets:
    phi,cc,res=analyze(P)
    d=np.array([math.degrees(math.acos(np.clip(np.dot(P0[sel][k],P[sel][k]),-1,1))) for k in range(len(sel))])
    h=ConvexHull(P); rg=math.degrees(max(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]]) for t in h.simplices))
    R[tag]=dict(phi_range=float(phi.max()-phi.min()),phi_std=float(phi.std()),
                phi_min=float(phi.min()),phi_max=float(phi.max()),
                res_median=float(np.median(res)),res_mean=float(res.mean()),
                res_max=float(res.max()),res_std=float(res.std()),
                rho_global=float(rg),disp_mean=float(d.mean()),disp_max=float(d.max()))
    R[tag]["_phi"]=phi; R[tag]["_cc"]=cc; R[tag]["_res"]=res

print(f"{'variant':<32} {'phi_range':>10} {'phi_std':>9} {'res_med':>9} {'res_max':>9} {'res_std':>9} {'rho_glob':>10}")
for tag,_ in sets:
    r=R[tag]
    print(f"{tag:<32} {r['phi_range']:>10.6f} {r['phi_std']:>9.6f} {r['res_median']:>9.6f} "
          f"{r['res_max']:>9.6f} {r['res_std']:>9.6f} {r['rho_global']:>10.6f}")

# 定位原版高残差点
# 注意：gen_meta 的 isedge 判据为 (i==0)or(j==0)or(j==i)，漏了 i==n 这条棱(BC 边)，
# 故改用"到三条棱的格点层数" d_edge = min(n-i, j, i-j) 作准确分类。
res=R["boundary-aware (paper 3.3.4)"]["_res"]
hi=np.where(res>0.05)[0]
print(f"\n=== 原版修正后 r_i>0.05deg 的点：{len(hi)} 个 ===")
def edge_layers(h):
    i,j=int(I[sel][h]),int(J[sel][h])
    return min(n-i,j,i-j)
for h in hi[:25]:
    print(f"  (I,J)=({I[sel][h]:2d},{J[sel][h]:2d})  r={res[h]:.6f}deg  "
          f"到最近棱格点层数={edge_layers(h)}")
layers=np.array([edge_layers(h) for h in hi])
print(f"  层数分布: 0层(恰在棱上)={int((layers==0).sum())}, 1层={int((layers==1).sum())}, "
      f"2层={int((layers==2).sum())}, >=3层={int((layers>=3).sum())}")
print(f"  -> 高残差点 {(layers<=2).sum()}/{len(hi)} 集中在距棱 2 层以内（边角区）")

# 三方残差对比图（共用 colorbar）
rvmax=max(R[t]["res_max"] for t,_ in sets)
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
for a,(tag,_) in zip(ax,sets):
    a.plot(tri2[:,0],tri2[:,1],'k-',lw=1.2); a.set_aspect('equal'); a.axis('off')
    r=R[tag]
    sc=a.scatter(P2[:,0],P2[:,1],c=r["_res"],s=14,cmap='viridis',vmin=0,vmax=rvmax)
    a.set_title(f'{tag}\nmedian={r["res_median"]:.5f} max={r["res_max"]:.5f} deg',fontsize=10)
    plt.colorbar(sc,ax=a,shrink=.8,label='r_i, deg')
fig.suptitle('1-ring balance residual: baseline vs two correction variants (same color scale)',fontsize=11)
fig.tight_layout(); fig.savefig("E:/毕业论文/算法/fig_residual_three_variants.png",dpi=145)
print("\n[出图] fig_residual_three_variants.png")

out={k:{kk:vv for kk,vv in r.items() if not kk.startswith('_')} for k,r in R.items()}
out["high_residual_count"]=int(len(hi))
out["high_residual_edge_layers"]={str(k):int(v) for k,v in zip(*np.unique(layers,return_counts=True))}
json.dump(out,open("E:/毕业论文/算法/compare_three_variants.json","w"),indent=2)
print("[完成] compare_three_variants.json")
