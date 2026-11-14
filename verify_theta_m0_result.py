"""
独立复核：联合优化最优点集 rho=1.488060 是否"无代价的改善"
=========================================================
1) 用 improve_corrected.rho（原版逐三角形，非向量化）直接读 CSV 点集重算 rho
2) 点集健康度：凸包顶点数、去重后点数(裂缝)、最小点间距
3) 均匀性/局部平衡：全局 Delaunay phi 的 std/range、1-ring 残差 max/median
4) 最坏三角形位置（dev / d_e）
5) 与 P0 / 方案C / 方案A / (m0=0)联合 的横向对比
"""
import numpy as np, math, csv, json
from scipy.spatial import cKDTree, ConvexHull
import improve_corrected as ic

OUT="E:/毕业论文/算法"; n=27
V=ic.V; faces=ic.faces
def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
Cs=np.array([normalize(V[f[0]]+V[f[1]]+V[f[2]]) for f in faces])

def read_pts(path):
    rows=list(csv.DictReader(open(path)))
    P0=np.array([[float(r['x']),float(r['y']),float(r['z'])] for r in rows])
    P =np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in rows])
    return P0,P

def circum_all(P):
    h=ConvexHull(P); t=h.simplices
    a0,b0,c0=P[t[:,0]],P[t[:,1]],P[t[:,2]]
    aa=np.arccos(np.clip(np.sum(a0*b0,1),-1,1))
    bb=np.arccos(np.clip(np.sum(b0*c0,1),-1,1))
    cc=np.arccos(np.clip(np.sum(c0*a0,1),-1,1))
    s=(aa+bb+cc)/2
    val=np.sin(s-aa)*np.sin(s-bb)*np.sin(s-cc)/np.sin(s)
    r=np.degrees(2*np.arctan(np.sqrt(np.clip(val,0,None))))
    return h,t,r

def dev_of(p):
    d=[math.degrees(math.acos(np.clip(np.dot(p,c),-1,1))) for c in Cs]
    return min(d)

def evaluate(name,path):
    P0,P=read_pts(path); N=len(P)
    rho_ref=ic.rho(P)                       # 原版逐三角形
    h,t,r=circum_all(P)
    hull_pts=len(np.unique(t.ravel()))
    uniq=len(np.unique(np.round(P,10),axis=0))
    dmin=math.degrees(float(cKDTree(P).query(P,k=2)[0][:,1].min()))
    rmax=np.full(N,-np.inf); rmin=np.full(N,np.inf)
    np.maximum.at(rmax,t.ravel(),np.repeat(r,3))
    np.minimum.at(rmin,t.ravel(),np.repeat(r,3))
    res=rmax-rmin
    worst=t[int(np.argmax(r))]
    ctr=normalize(P[worst].sum(axis=0))
    disp=np.degrees(np.arccos(np.clip(np.sum(P0*P,axis=1),-1,1)))
    return dict(name=name,n=N,rho=float(rho_ref),rho_vec=float(r.max()),
                hull_pts=hull_pts,unique_pts=uniq,min_sep_deg=dmin,
                phi_std=float(r.std()),phi_range=float(r.ptp()),
                res_max=float(res.max()),res_median=float(np.median(res)),
                worst_tri=sorted(int(q) for q in worst),
                worst_dev_deg=float(dev_of(ctr)),
                disp_mean=float(disp.mean()),disp_max=float(disp.max()))

CASES=[
 ("P0 (baseline)",      "corrected_points_n27.csv"),          # 注意: 这是原版修正! 用 P0 列重建
 ("scheme A radial",    "symmetric_points_n27.csv"),
 ("scheme C (correct)", "symmetric_points_final_C.csv"),
 ("joint m0=0",         "theta_joint_points_n27.csv"),
 ("joint m0 free BEST", "theta_m0_joint_points_n27.csv"),
]
rows=list(csv.DictReader(open(OUT+"/corrected_points_n27.csv")))
P0=np.array([[float(q['x']),float(q['y']),float(q['z'])] for q in rows])
# 造一个真正的 P0 基线 CSV 供统一口径
import os
if not os.path.exists(OUT+"/baseline_P0_n27.csv"):
    with open(OUT+"/baseline_P0_n27.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["x","y","z","x_opt","y_opt","z_opt"])
        for p in P0: w.writerow([p[0],p[1],p[2],p[0],p[1],p[2]])
CASES[0]=("P0 (baseline)","baseline_P0_n27.csv")

print("="*118)
hdr=["name","rho","hull","uniq","min_sep","phi_std","phi_range","res_max","res_med","worst_dev","disp_mean","disp_max"]
print(f"{'name':22s} {'rho':>9s} {'hull':>6s} {'uniq':>6s} {'min_sep':>9s} {'phi_std':>9s} "
      f"{'phi_rng':>9s} {'res_max':>9s} {'res_med':>9s} {'w_dev':>8s} {'d_mean':>8s} {'d_max':>8s}")
out={}
for name,path in CASES:
    e=evaluate(name,OUT+"/"+path); out[name]=e
    print(f"{name:22s} {e['rho']:9.6f} {e['hull_pts']:6d} {e['unique_pts']:6d} {e['min_sep_deg']:9.6f} "
          f"{e['phi_std']:9.6f} {e['phi_range']:9.6f} {e['res_max']:9.6f} {e['res_median']:9.6f} "
          f"{e['worst_dev_deg']:8.3f} {e['disp_mean']:8.6f} {e['disp_max']:8.6f}")
print("="*118)

best=out["joint m0 free BEST"]; c=out["scheme C (correct)"]
print("\n=== 结论性对比 ===")
print(f"  rho: 方案C={c['rho']:.6f} -> 联合最优={best['rho']:.6f}  "
      f"改善={c['rho']-best['rho']:+.6f} deg")
for k,label in [('phi_std','phi 标准差'),('phi_range','phi 极差'),
                ('res_max','残差 max'),('res_median','残差 median'),
                ('min_sep_deg','最小点间距'),('disp_mean','平均位移')]:
    print(f"  {label:12s}: {c[k]:.6f} -> {best[k]:.6f}  ({best[k]-c[k]:+.6f})")
print(f"  点集健康: 凸包顶点 {best['hull_pts']}/7292, 去重 {best['unique_pts']}/7292, "
      f"最坏三角形重心 dev={best['worst_dev_deg']:.3f}deg")
assert best['hull_pts']==7292 and best['unique_pts']==7292, "点集退化或有裂缝"
assert abs(best['rho']-best['rho_vec'])<1e-8, "两套 rho 不一致"
print("\n[独立复核通过] rho 由原版逐三角形算法确认，点集无退化/无裂缝")
# ---------- 从最终点集反算"偏转角 vs 到棱距离"曲线 ----------
print("\n=== 边界条件验证：实际偏转角 psi(相对朝面心径向) 随 d_e 的变化 ===")
Pg,fi,I,J,isedge=ic.gen_meta(n)
Cg,devg,isv,ue,mb,Cs2=ic.build_fields(Pg,fi)
UR=np.array([normalize(Cg[k]-np.dot(Cg[k],P0[k])*P0[k]) for k in range(len(P0))])

def psi_profile(path):
    rows=list(csv.DictReader(open(path)))
    q0=np.array([[float(r['x']),float(r['y']),float(r['z'])] for r in rows])
    q1=np.array([[float(r['x_opt']),float(r['y_opt']),float(r['z_opt'])] for r in rows])
    de=np.array([float(r['d_e_deg']) for r in rows])
    th_col=np.array([float(r['theta_deg']) for r in rows])
    d=q1-q0; d=d-np.sum(d*q0,axis=1)[:,None]*q0
    cosp=np.sum(d*UR,axis=1)/(np.linalg.norm(d,axis=1)+1e-30)
    psi=np.degrees(np.arccos(np.clip(np.abs(cosp),-1,1)))
    ok=np.linalg.norm(d,axis=1)>1e-9
    return de,th_col,psi,ok

path_best=OUT+"/theta_m0_joint_points_n27.csv"
de,th_col,psi,ok=psi_profile(path_best)
bins=[0,0.01,0.5,1.5,3,5,8,12,21.5]
prof=[]
print(f"  {'d_e 区间(deg)':>16s} {'点数':>6s} {'设计 theta':>12s} {'实测 psi':>10s}")
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(de>=lo)&(de<hi)&ok
    if m.sum()==0: continue
    row=dict(lo=lo,hi=hi,n=int(m.sum()),theta_mean=float(th_col[m].mean()),
             psi_mean=float(psi[m].mean()),psi_median=float(np.median(psi[m])))
    prof.append(row)
    print(f"  [{lo:5.2f},{hi:5.2f}) {m.sum():6d} {row['theta_mean']:12.2f} {row['psi_mean']:10.2f}")
print(f"  -> 棱上(d_e=0) psi={prof[0]['psi_mean']:.2f}deg, 远离棱 psi={prof[-1]['psi_mean']:.2f}deg")
out["_boundary_profile"]=prof
out["_boundary_note"]="psi measured from final point set; theta is the designed (pre-Reynolds) value"
json.dump(out,open(OUT+"/verify_theta_m0_result.json","w"),indent=2)
print("[交付] verify_theta_m0_result.json")
