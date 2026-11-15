"""
独立复核 deepseek 交叉核验结论中【可数值验证】的条目
=====================================================
逐条实算，不采信任何转述数字：
  A. tan R >= 2 tan r 是否成立、R >= 2r 是否不成立、等边情形
  B. 旋转轴上的格点数（12 顶点 + 20 面心？）、棱内点数与非零位移
  C. min_bary=0.08 对应的 d_e 是多少
  D. 未做 G120 群平均的初步场（用公布系数）的对称偏差
  E. Fejes 下界随 M：n=26 的下界、下界=1.5° 的临界 M
  F. 欧拉缺陷计数 Σ(6-k)=12
"""
import numpy as np, math, csv, sys, json
sys.path.insert(0, "E:/毕业论文/算法")
import improve_corrected as ic
from covering_radius import sph_circumradius, rho
from scipy.spatial import cKDTree, ConvexHull

OUT = "E:/毕业论文/算法"; n = 27
def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0
def rot_axis(k,t):
    k=normalize(k); c,s=math.cos(t),math.sin(t)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

P0,fi,I,J,ie = ic.gen_meta(n)
C,dev,isv,_,mb,Cs = ic.build_fields(P0,fi)
N=len(P0); V=ic.V; faces=ic.faces
u_rad = np.array([normalize(C[k]-np.dot(C[k],P0[k])*P0[k]) for k in range(N)])
d_e = np.zeros(N); u_tan = np.zeros((N,3))
for k in range(N):
    f=faces[fi[k]]; A,B,Cc=V[f[0]],V[f[1]],V[f[2]]
    best=None; bd=1e9
    for (X,Y) in [(A,B),(B,Cc),(Cc,A)]:
        ne=normalize(np.cross(X,Y))
        dd=math.degrees(math.asin(min(1.,abs(np.dot(P0[k],ne)))))
        if dd<bd: bd=dd; best=(X,Y,ne)
    X,Y,ne=best; d_e[k]=bd
    ue=normalize(np.cross(P0[k],ne))
    s=ic.angle(P0[k],X)-ic.angle(P0[k],Y)
    u_tan[k]=(1. if s>=0 else -1.)*ue

# ---------- A ----------
t=ConvexHull(P0).simplices
R=np.zeros(len(t)); r=np.zeros(len(t))
for q,(i,j,k) in enumerate(t):
    a,b,c=P0[i],P0[j],P0[k]
    A1,B1,C1=ic.angle(a,b),ic.angle(b,c),ic.angle(c,a); s=(A1+B1+C1)/2
    val=math.sin(s-A1)*math.sin(s-B1)*math.sin(s-C1)/math.sin(s)
    r[q]=math.atan(math.sqrt(max(0.,val)))
    R[q]=sph_circumradius(a,b,c)
Rdeg,rdeg=np.degrees(R),np.degrees(r)
viol_2r=int((R < 2*r-1e-14).sum())
viol_tan=int((np.tan(R) < 2*np.tan(r)-1e-14).sum())
print("=== A. 球面欧拉不等式 ===")
print(f"  R < 2r 的三角形数 = {viol_2r}/{len(t)}   max(2r-R)={np.max(2*rdeg-Rdeg):.9f}deg")
print(f"  tanR < 2tanr 的三角形数 = {viol_tan}/{len(t)}")
print(f"  mean(R-2r)={np.mean(Rdeg-2*rdeg):.6f}deg  max|R-2r|={np.max(np.abs(Rdeg-2*rdeg)):.6f}deg")
# 等边对照
alpha=math.radians(2.)
a=np.array([1.,0,0]); b=np.array([math.cos(alpha),math.sin(alpha),0])
x=math.cos(alpha)*(1-math.cos(alpha))/math.sin(alpha); y=math.sqrt(math.sin(alpha)**2-x*x)
c=np.array([math.cos(alpha),x,y])
A1=B1=C1=alpha; s=3*alpha/2
val=math.sin(s-A1)*math.sin(s-B1)*math.sin(s-C1)/math.sin(s)
rr=math.atan(math.sqrt(val)); RR=sph_circumradius(a,b,c)
print(f"  等边(a=b=c=2deg): R={np.degrees(RR):.6f} 2r={np.degrees(2*rr):.6f} "
      f"R-2r={np.degrees(RR-2*rr):.9f}  tanR-2tanr={math.tan(RR)-2*math.tan(rr):.3e}")

# ---------- B ----------
Cs_arr=np.array([normalize(V[f[0]]+V[f[1]]+V[f[2]]) for f in faces])
face_center_hits=int(sum(1 for k in range(N) if min(ic.angle(P0[k],cc) for cc in Cs_arr)<1e-6))
# 棱中点（n 奇数应无格点）
mids=[]
for f in faces:
    for (p,q) in [(V[f[0]],V[f[1]]),(V[f[1]],V[f[2]]),(V[f[2]],V[f[0]])]:
        mids.append(normalize(p+q))
mid_hits=int(sum(1 for k in range(N) if min(ic.angle(P0[k],m) for m in mids)<1e-6))
rows=list(csv.DictReader(open(f"{OUT}/correct_points_J.csv")))
disp=np.array([float(q['disp_deg']) for q in rows])
de_csv=np.array([float(q['d_e_deg']) for q in rows])
edge_inner=(de_csv<0.01)&(disp>1e-9)
print("\n=== B. 对称轴点计数 ===")
print(f"  顶点={isv.sum()}  面心格点={face_center_hits}  棱中点格点={mid_hits}")
print(f"  => 旋转轴上格点 = {isv.sum()+face_center_hits+mid_hits}（5重轴+3重轴+2重轴）")
print(f"  共享棱内点(tie, d_e≈0, 非顶点) = {edge_inner.sum()}  "
      f"位移 min/median/max = {disp[edge_inner].min():.6f}/{np.median(disp[edge_inner]):.6f}/{disp[edge_inner].max():.6f} deg")
print(f"  792 = 30棱×26内点 + 12顶点 = {30*26+12}（是共享棱 tie 点，不是旋转轴点）")

# ---------- C ----------
sel=(mb>=0.075)&(mb<=0.085)
print("\n=== C. min_bary=0.08 对应 d_e ===")
print(f"  mb∈[0.075,0.085] 点数={sel.sum()}  d_e mean={d_e[sel].mean():.4f} "
      f"min={d_e[sel].min():.4f} max={d_e[sel].max():.4f} deg")
sel2=(d_e>=4.6)&(d_e<=5.0)
print(f"  d_e∈[4.6,5.0] 点数={sel2.sum()}  mb mean={mb[sel2].mean():.4f}")

# ---------- D ----------
av=normalize(V[1]); af=normalize(V[0]+V[4]+V[8])
gens=[rot_axis(av,2*math.pi/5),rot_axis(av,-2*math.pi/5),
      rot_axis(af,2*math.pi/3),rot_axis(af,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M=todo.pop(); key=tuple(np.round(M.ravel(),8))
    if key in seen: continue
    seen.add(key); G60.append(M)
    for g in gens: todo.append(g@M)
G120=G60+[-g for g in G60]
def sym_dev(P):
    tr=cKDTree(P); worst=0.
    for g in G120:
        d,_=tr.query((g@P.T).T); worst=max(worst,float(d.max()))
    return math.degrees(worst)
def expmap(d0):
    return np.array([normalize(math.cos(np.radians(np.linalg.norm(d0[k])))*P0[k]
                    + math.sin(np.radians(np.linalg.norm(d0[k])))*normalize(d0[k]))
                    if np.linalg.norm(d0[k])>1e-12 else P0[k] for k in range(N)])
# L
wL=np.clip((0.08-mb)/0.08,0,1)
dirL=np.array([normalize((1-wL[k])*u_rad[k]+wL[k]*u_tan[k]) for k in range(N)])
d0L=(0.010588+0.001158*dev)[:,None]*dirL; d0L[isv]=0
# J
wJ=np.clip(1-d_e/10.502761,0,1); th=(math.pi/2)*wJ
d0J=(0.035363+0.000876*dev)[:,None]*(np.cos(th)[:,None]*u_rad+np.sin(th)[:,None]*u_tan); d0J[isv]=0
print("\n=== D. 未群平均初步场的对称偏差 ===")
print(f"  P0 基线={sym_dev(P0):.9f}deg  方案L初步场={sym_dev(expmap(d0L)):.6f}deg  "
      f"方案J初步场={sym_dev(expmap(d0J)):.6f}deg")

# ---------- E ----------
def fejes(M):
    w=M*math.pi/(6*M-12)
    return math.degrees(math.acos((1/math.sqrt(3))*math.cos(w)/math.sin(w)))
M26=10*26*26+2; M27=10*27*27+2
lo,hi=100,20000
for _ in range(80):
    mid=(lo+hi)/2
    if fejes(mid)>1.5: lo=mid
    else: hi=mid
print("\n=== E. Fejes 下界与最小阶数 ===")
print(f"  M={M26} (n=26): 下界={fejes(M26):.6f}deg >1.5 ⇒ 不可能达标")
print(f"  M={M27} (n=27): 下界={fejes(M27):.6f}deg <1.5")
print(f"  下界恰好=1.5deg 的临界 M*≈{hi:.1f}；M<{hi:.0f} 的任何点集都不可能满足 ρ≤1.5°")

# ---------- F ----------
print("\n=== F. 欧拉缺陷计数 Σ(6-k)=12 ===")
for name,k,cnt in [('正四面体',3,4),('正八面体',4,6),('正二十面体',5,12)]:
    print(f"  {name}: 顶点度 k={k}, 顶点数={cnt}, Σ(6-k)={cnt*(6-k)}")

json.dump(dict(A=dict(R_lt_2r_count=viol_2r,tanR_lt_2tanr_count=viol_tan,
                      mean_R_minus_2r_deg=float(np.mean(Rdeg-2*rdeg)),
                      equilateral_R_deg=float(np.degrees(RR)),equilateral_2r_deg=float(np.degrees(2*rr))),
               B=dict(vertices=int(isv.sum()),face_centers=face_center_hits,edge_midpoints=mid_hits,
                      axis_points=int(isv.sum()+face_center_hits+mid_hits),
                      edge_inner_points=int(edge_inner.sum()),
                      edge_disp_mean_deg=float(disp[edge_inner].mean())),
               C=dict(d_e_at_mb_008_deg=float(d_e[sel].mean()),mb_at_d_e_48_deg=float(mb[sel2].mean())),
               D=dict(P0=sym_dev(P0),L_unprojected=sym_dev(expmap(d0L)),J_unprojected=sym_dev(expmap(d0J))),
               E=dict(M_n26=M26,lower_n26_deg=fejes(M26),M_n27=M27,lower_n27_deg=fejes(M27),
                      critical_M_for_1p5=hi)),
          open(f"{OUT}/verify_deepseek_claims.json","w"),indent=2)
print("\n[交付] verify_deepseek_claims.json")
