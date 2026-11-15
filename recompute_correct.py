"""
用【正确的球面外接圆半径】重算位置拟合优化两方案的系数
========================================================
2026-11-15。算法框架完全保留（1/6 局部几何量 → 初步位移场 → G120 Reynolds
严格对称化 → ExpMap → 全局 Delaunay 校验），只把目标函数从错误的
max(2r) 换成正确的 max(R)。

  方案 L 线性位移版：w = clip((0.08 - mb)/0.08,0,1)，线性混合 + 逐点归一化，
                     m(dev)=m0+a*dev，变量 (m0,a)
  方案 J 角度联合版：w = clip(1 - d_e/lam,0,1)，theta=(pi/2)w，
                     m(dev)=m0+a*dev，变量 (m0,a,lam)

加速：Reynolds 线性 + alpha/beta 为群不变标量 ⇒ 预计算 Arad/Atan 两个固定场。
脚本内做不变量自检，并对最优解再用【完整 Reynolds】逐点复核。

输出：correct_points_L.csv / correct_points_J.csv、correct_coef_*.json、
      recompute_correct.json（含门禁与 face0 均匀性指标）、v_correct_*.npy
"""
import numpy as np, math, json, csv, time, sys
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize
import improve_corrected as ic
from covering_radius import rho as rho_correct, sph_circumradius

OUT = "E:/毕业论文/算法"; n = 27
t0 = time.time()
np.set_printoptions(precision=6, suppress=True)

def normalize(v):
    v = np.asarray(v, float); s = np.linalg.norm(v); return v/s if s > 1e-15 else v*0.0

def rot_axis(k, th):
    k = normalize(k); c, s = math.cos(th), math.sin(th)
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c + s*K + (1-c)*np.outer(k,k)

# ---------------- 1. 几何量 ----------------
P0, fi, I, J, isedge = ic.gen_meta(n)
C, dev, is_vertex, _, min_bary, Cs = ic.build_fields(P0, fi)
N = len(P0); V = ic.V; faces = ic.faces
assert N == 7292
u_rad = np.array([normalize(C[k]-np.dot(C[k],P0[k])*P0[k]) for k in range(N)])
d_e = np.zeros(N); u_tan = np.zeros((N,3))
for k in range(N):
    f = faces[fi[k]]; A,B,Cc = V[f[0]],V[f[1]],V[f[2]]
    best = None; bd = 1e9
    for (X,Y) in [(A,B),(B,Cc),(Cc,A)]:
        ne = normalize(np.cross(X,Y))
        dd = math.degrees(math.asin(min(1.0, abs(np.dot(P0[k],ne)))))
        if dd < bd: bd = dd; best = (X,Y,ne)
    X,Y,ne = best; d_e[k] = bd
    ue = normalize(np.cross(P0[k],ne))
    s = ic.angle(P0[k],X) - ic.angle(P0[k],Y)
    u_tan[k] = (1.0 if s >= 0 else -1.0)*ue
print(f"[几何] N={N} dev∈[{dev.min():.2f},{dev.max():.2f}] d_e∈[{d_e.min():.3f},{d_e.max():.2f}]")

# 方案 L 的固定权重与归一化因子
wL = np.clip((0.08 - min_bary)/0.08, 0.0, 1.0)
mixL = (1-wL)[:,None]*u_rad + wL[:,None]*u_tan
sL = np.linalg.norm(mixL, axis=1); sL = np.where(sL < 1e-12, 1.0, sL)

# ---------------- 2. 群 G120 ----------------
av = normalize(V[1]); af = normalize(V[0]+V[4]+V[8])
gens = [rot_axis(av,2*math.pi/5), rot_axis(av,-2*math.pi/5),
        rot_axis(af,2*math.pi/3), rot_axis(af,-2*math.pi/3)]
G60=[]; seen=set(); todo=[np.eye(3)]
while todo:
    M = todo.pop(); key = tuple(np.round(M.ravel(),8))
    if key in seen: continue
    seen.add(key); G60.append(M)
    for g in gens: todo.append(g@M)
G120 = G60 + [-g for g in G60]
IDX = [cKDTree(P0).query((g@P0.T).T)[1] for g in G120]
print(f"[群] |G120|={len(G120)}")

# ---------------- 3. 标量群不变性自检 ----------------
scalars = dict(dev=dev, min_bary=min_bary, d_e=d_e, sL=sL)
inv = {k: max(float(np.abs(v[j]-v).max()) for j in IDX) for k,v in scalars.items()}
print("[自检①] 标量群不变性最大偏差:", {k: f"{v:.3e}" for k,v in inv.items()})
assert max(inv.values()) < 1e-6, "存在非群不变标量，线性分解不成立"

# ---------------- 4. Reynolds 线性分解预计算 ----------------
Arad = np.zeros((N,3)); Atan = np.zeros((N,3))
for g,j in zip(G120, IDX):
    Arad += (g.T@u_rad[j].T).T
    Atan += (g.T@u_tan[j].T).T
Arad /= len(G120); Atan /= len(G120)

def reynolds_full(d0):
    acc = np.zeros((N,3))
    for g,j in zip(G120, IDX): acc += (g.T@d0[j].T).T
    return acc/len(G120)

# 自检②：用真实形式的 alpha/beta 比较快速分解与完整 Reynolds
al_t = (0.003*dev)*(1-wL)/sL; be_t = (0.003*dev)*wL/sL
v_full = reynolds_full(al_t[:,None]*u_rad + be_t[:,None]*u_tan)
v_fast = al_t[:,None]*Arad + be_t[:,None]*Atan
err_dec = float(np.abs(v_full-v_fast).max())
print(f"[自检②] 线性分解 vs 完整 Reynolds 最大差={err_dec:.3e}")
assert err_dec < 1e-8

def build_P(alpha, beta, v_override=None):
    v = alpha[:,None]*Arad + beta[:,None]*Atan if v_override is None else v_override
    mag = np.linalg.norm(v, axis=1)
    u = np.zeros_like(v); nz = mag > 1e-12
    u[nz] = v[nz]/mag[nz,None]
    t = np.radians(mag)[:,None]
    P = np.cos(t)*P0 + np.sin(t)*u
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P[is_vertex] = P0[is_vertex]
    return P, v

# ---------------- 5. 目标函数与两方案场 ----------------
def field_L(z):
    m0, a = np.exp(z)
    m = m0 + a*dev
    return m*(1-wL)/sL, m*wL/sL, None

def field_J(z):
    m0, a, lam = np.exp(z)
    w = np.clip(1 - d_e/lam, 0, 1)
    th = (math.pi/2)*w
    m = m0 + a*dev
    return m*np.cos(th), m*np.sin(th), dict(w=w, theta=th, m=m)

def objective(z, field):
    al, be, _ = field(z)
    P,_ = build_P(al, be)
    return rho_correct(P)

STARTS_L = [[0.001,0.004],[0.02,0.003],[0.05,0.001],[0.0001,0.006]]
STARTS_J = [[0.001,0.004,5],[0.02,0.003,3],[0.05,0.001,8],[0.001,0.006,12]]

def fit(field, starts, label):
    best = None
    for s0 in starts:
        z0 = np.log(np.array(s0,float)); k = len(z0)
        simp = np.array([z0]+[z0+np.eye(k)[i]*0.08 for i in range(k)])
        r = minimize(objective, z0, args=(field,), method='Nelder-Mead',
                     options={'maxiter':420,'initial_simplex':simp,'xatol':1e-6,'fatol':1e-9})
        print(f"  {label} start={s0} -> rho={r.fun:.6f} nfev={r.nfev}", flush=True)
        if best is None or r.fun < best[0]: best = (float(r.fun), np.exp(r.x))
    return best

print("\n=== 方案 L 拟合（正确外接圆半径） ===", flush=True)
bL = fit(field_L, STARTS_L, "L")
print("=== 方案 J 拟合（正确外接圆半径） ===", flush=True)
bJ = fit(field_J, STARTS_J, "J")

# ---------------- 6. 门禁 ----------------
def strict_sym(P, tol=1e-9):
    t = cKDTree(P); bad = 0; worst = 0.0
    for g in G120:
        d,j = t.query((g@P.T).T)
        if len(np.unique(j)) != len(P) or d.max() > tol: bad += 1
        worst = max(worst, float(d.max()))
    return bad, math.degrees(worst)

def mirror_median(P):
    A,B,Cc = V[0],V[4],V[8]; G = normalize(A+B+Cc)
    nA = normalize(np.cross(G,A)); S = np.eye(3)-2*np.outer(nA,nA)
    sel = np.where(fi==0)[0]; base = P0[sel]
    jj = cKDTree(base).query((S@base.T).T)[1]
    def td(Z):
        d = Z[sel]-base; return d-np.sum(d*base,axis=1)[:,None]*base
    d = td(P); Sd = (S@d.T).T
    num = np.sum(Sd*d[jj],axis=1)
    den = np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[jj],axis=1)+1e-30
    return float(np.median(np.degrees(np.arccos(np.clip(num/den,-1,1)))))

def edge_signed(P):
    sel = np.where(fi==0)[0]; base = P0[sel]
    d = P[sel]-base; d = d-np.sum(d*base,axis=1)[:,None]*base
    A,B,Cc = V[0],V[4],V[8]; out=[]
    for cond,u in [(lambda i,j:j==0,normalize(B-A)),(lambda i,j:j==i,normalize(Cc-A)),
                   (lambda i,j:i==n,normalize(Cc-B))]:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        pr=np.array([np.dot(d[h],u) for h in ks])
        out.append(float(pr.mean()))
    return out

# face 0 构造三角形（729 个）上的 phi / 1-ring 残差，用正确外接圆半径
sel0 = np.where(fi==0)[0]
loc = {(int(I[k]),int(J[k])):h for h,k in enumerate(sel0)}
tris=[]
for i in range(n):
    for j in range(i+1):
        tris.append((loc[(i,j)],loc[(i+1,j)],loc[(i+1,j+1)]))
        if j<i: tris.append((loc[(i,j)],loc[(i+1,j+1)],loc[(i,j+1)]))
tris=np.array(tris)
def face0_metrics(P):
    Pf = P[sel0]
    phi = np.array([math.degrees(sph_circumradius(Pf[t[0]],Pf[t[1]],Pf[t[2]])) for t in tris])
    ring=[[] for _ in range(len(sel0))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    res=np.array([max(a)-min(a) for a in ring])
    return dict(phi_std=float(phi.std()),phi_range=float(np.ptp(phi)),
                res_max=float(res.max()),res_median=float(np.median(res)))

results={}
for label,best,field in [("L",bL,field_L),("J",bJ,field_J)]:
    al,be,extra = field(np.log(best[1]))
    d0 = al[:,None]*u_rad + be[:,None]*u_tan
    d0[is_vertex]=0.0
    v_full = reynolds_full(d0)
    P_fast,_ = build_P(al,be)
    P_full,_ = build_P(al,be,v_override=v_full)
    r_fast, r_full = rho_correct(P_fast), rho_correct(P_full)
    P = P_full; r = r_full
    bad,worst = strict_sym(P); mir = mirror_median(P); es = edge_signed(P)
    uniq = len(np.unique(np.round(P,10),axis=0))
    disp = np.degrees(np.arccos(np.clip(np.sum(P0*P,axis=1),-1,1)))
    fm = face0_metrics(P)
    ok = (len(P)==7292 and bad==0 and r<=1.5 and mir<1e-3 and
          max(abs(x) for x in es)<1e-6 and uniq==7292)
    coef = list(map(float,best[1]))
    results[label] = dict(rho=r, rho_fast=r_fast, params=coef,
                          param_names=["m0","a"] if label=="L" else ["m0","a","lam"],
                          strict_sym_bad=bad, sym_worst_deg=worst, mirror_median_deg=mir,
                          edge_signed_mean=es, unique_points=uniq,
                          disp_mean_deg=float(disp.mean()), disp_max_deg=float(disp.max()),
                          face0=fm, feasible_le_1p5=bool(r<=1.5), gate_all_pass=bool(ok),
                          fast_vs_full_point_diff=float(np.abs(P_fast-P_full).max()))
    print(f"\n[{label}] rho={r:.6f} (fast {r_fast:.6f}, 点集差 {results[label]['fast_vs_full_point_diff']:.2e}) "
          f"params={np.round(coef,7)}  <=1.5? {r<=1.5}")
    print(f"      对称失败={bad}/120 偏差={worst:.9f} 镜像={mir:.6f} 去重={uniq} "
          f"位移均值={disp.mean():.6f} max={disp.max():.6f}")
    print(f"      face0: {fm}")
    np.save(f"{OUT}/v_correct_{label}.npy", v_full)
    with open(f"{OUT}/correct_points_{label}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j",
                                     "dev_deg","d_e_deg","disp_deg","is_vertex"])
        for k in range(N):
            w.writerow([P0[k,0],P0[k,1],P0[k,2],P[k,0],P[k,1],P[k,2],
                        int(fi[k]),int(I[k]),int(J[k]),dev[k],d_e[k],disp[k],int(is_vertex[k])])
    json.dump(dict(variant=f"correct_radius_scheme_{label}", rho_max=r, **results[label]),
              open(f"{OUT}/correct_coef_{label}.json","w"),indent=2)
    print(f"      门禁: {'全部通过' if ok else '⚠️ 有未通过项'}")

baseline = dict(P0=rho_correct(P0), fejes_lower=1.47569240)
print(f"\n=== 基线（正确口径） P0={baseline['P0']:.6f}deg  Fejes下界={baseline['fejes_lower']:.6f}deg ===")
report = dict(date="2026-11-15", radius_formula="GEOM: R=arccos(|n.a|), n=normalize((b-a)x(c-a))",
              note="全部系数在【正确外接圆半径】目标下重新拟合；算法框架与 G120 严格对称化不变",
              baseline=baseline, selfcheck=dict(scalar_invariance=inv, decomp_err=err_dec),
              schemes=results, elapsed_sec=time.time()-t0)
json.dump(report, open(f"{OUT}/recompute_correct.json","w"),indent=2)
print(f"[交付] recompute_correct.json / correct_points_L.csv / correct_points_J.csv / correct_coef_*.json")
print(f"[完成] 用时 {time.time()-t0:.1f}s")
