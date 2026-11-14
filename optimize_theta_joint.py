"""
联合优化实验：偏转函数 theta(.) 与位移比例 a
=============================================
在方案 C（严格 120 阶对称 + 协变切向边界感知）框架下，把写死的
wb = clip((0.08-min_bary)/0.08) 换成可优化函数 W(d_e)，并与
m(dev) = a*dev 的比例系数 a 做联合 Nelder-Mead 优化，目标 min rho_max。

保持的核心性质：
  - 12 个顶点固定
  - 棱上 theta=90deg（纯沿棱）、面心 theta=0（纯向心）
  - 切向乘关于棱中点的奇函数 sign(angle(p,X)-angle(p,Y)) 保证协变
  - G120 Reynolds 投影 => 严格 120 阶对称

计算加速：Reynolds 是线性算子，且 alpha=m*cos(theta)、beta=m*sin(theta)
是标量群不变量，故预计算 Arad/Atan 两个固定场，每次评估只需 O(N) 乘加。
脚本内做三项自检，并对最优解用【完整 Reynolds】复核。
"""
import numpy as np, math, json, csv, os, time
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize
import improve_corrected as ic

OUT = "E:/毕业论文/算法"
n = 27
np.set_printoptions(precision=6, suppress=True)

def normalize(v):
    v = np.asarray(v, float); s = np.linalg.norm(v)
    return v/s if s > 1e-15 else v*0.0

def rot_axis(k, th):
    k = normalize(k); c = math.cos(th); s = math.sin(th)
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c + s*K + (1-c)*np.outer(k,k)

# ================= 1. 几何量 =================
t_start = time.time()
P0, fi, I, J, isedge = ic.gen_meta(n)
C, dev, is_vertex, u_edge_old, min_bary, Cs = ic.build_fields(P0, fi)
N = len(P0); V = ic.V; faces = ic.faces
assert N == 10*n*n + 2 == 7292, N

u_rad = np.array([normalize(C[k]-np.dot(C[k],P0[k])*P0[k]) for k in range(N)])
d_e = np.zeros(N); u_tan = np.zeros((N,3))
for k in range(N):
    f = faces[fi[k]]; A,B,Cc = V[f[0]], V[f[1]], V[f[2]]
    edges = [(A,B),(B,Cc),(Cc,A)]
    best = None; bd = 1e9
    for (X,Y) in edges:
        ne = normalize(np.cross(X,Y))
        dd = math.degrees(math.asin(min(1.0, abs(np.dot(P0[k],ne)))))
        if dd < bd: bd = dd; best = (X,Y,ne)
    X,Y,ne = best
    d_e[k] = bd
    ue = normalize(np.cross(P0[k],ne))
    s = ic.angle(P0[k],X) - ic.angle(P0[k],Y)       # 关于棱中点的奇函数
    u_tan[k] = (1.0 if s >= 0 else -1.0)*ue
print(f"[几何] N={N} 顶点={is_vertex.sum()}  "
      f"dev∈[{dev.min():.3f},{dev.max():.3f}]deg  d_e∈[{d_e.min():.6f},{d_e.max():.3f}]deg")
print(f"       d_e 分位数(10/25/50/75/90%)={np.round(np.percentile(d_e,[10,25,50,75,90]),3)}")

# ================= 2. 群 G120 =================
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
tree = cKDTree(P0)
IDX = [tree.query((g@P0.T).T)[1] for g in G120]
print(f"[群] |G120|={len(G120)}")

# ================= 3. 自检①：标量群不变性 =================
inv_dev = max(float(np.abs(dev[j]-dev).max()) for j in IDX)
inv_de  = max(float(np.abs(d_e[j]-d_e).max()) for j in IDX)
print(f"[自检①] 群不变性: dev 最大偏差={inv_dev:.3e}deg  d_e 最大偏差={inv_de:.3e}deg")
assert inv_dev < 1e-6 and inv_de < 1e-6, "标量场不是群不变量，线性分解不成立"

# ================= 4. Reynolds 线性分解 =================
Arad = np.zeros((N,3)); Atan = np.zeros((N,3))
for g,j in zip(G120, IDX):
    Arad += (g.T@u_rad[j].T).T
    Atan += (g.T@u_tan[j].T).T
Arad /= len(G120); Atan /= len(G120)

def reynolds_full(d0):
    acc = np.zeros((N,3))
    for g,j in zip(G120, IDX): acc += (g.T@d0[j].T).T
    return acc/len(G120)

# 自检②：快速分解 vs 完整 Reynolds
# 注意：分解仅在 alpha/beta 为【群不变量标量场】时成立，故必须用真实形式
#       alpha=a*dev*cos(theta(d_e)), beta=a*dev*sin(theta(d_e))，不能用随机数。
def alpha_beta(a, w):
    th = (math.pi/2)*np.clip(w,0,1)
    return a*dev*np.cos(th), a*dev*np.sin(th)

al_t, be_t = alpha_beta(0.004, np.exp(-d_e/5.0))
inv_a = max(float(np.abs(al_t[j]-al_t).max()) for j in IDX)
inv_b = max(float(np.abs(be_t[j]-be_t).max()) for j in IDX)
v_full = reynolds_full(al_t[:,None]*u_rad + be_t[:,None]*u_tan)
v_fast = al_t[:,None]*Arad + be_t[:,None]*Atan
err_dec = float(np.abs(v_full-v_fast).max())
print(f"[自检②] alpha/beta 群不变性偏差={inv_a:.3e}/{inv_b:.3e}  "
      f"线性分解 vs 完整 Reynolds 最大差={err_dec:.3e}")
assert max(inv_a,inv_b) < 1e-9 and err_dec < 1e-9, "线性分解错误"

# ================= 5. 快速 rho =================
def rho_fast(P):
    h = ConvexHull(P); t = h.simplices
    a0,b0,c0 = P[t[:,0]], P[t[:,1]], P[t[:,2]]
    aa = np.arccos(np.clip(np.sum(a0*b0,1),-1,1))
    bb = np.arccos(np.clip(np.sum(b0*c0,1),-1,1))
    cc = np.arccos(np.clip(np.sum(c0*a0,1),-1,1))
    s = (aa+bb+cc)/2
    val = np.sin(s-aa)*np.sin(s-bb)*np.sin(s-cc)/np.sin(s)
    return math.degrees(float((2*np.arctan(np.sqrt(np.clip(val,0,None)))).max()))

# 自检③：向量化 rho vs 原版 rho
r_ref = ic.rho(P0); r_new = rho_fast(P0)
print(f"[自检③] rho(P0): 原版={r_ref:.9f}  向量化={r_new:.9f}  差={abs(r_ref-r_new):.3e}")
assert abs(r_ref-r_new) < 1e-9, "向量化 rho 不一致"

# ================= 6. 场构造 =================
def build_P(a, w, v_override=None):
    th = (math.pi/2)*np.clip(w,0,1)
    al = a*dev*np.cos(th); be = a*dev*np.sin(th)
    v = al[:,None]*Arad + be[:,None]*Atan if v_override is None else v_override
    mag = np.linalg.norm(v,axis=1)
    u = np.zeros_like(v); nz = mag > 1e-12
    u[nz] = v[nz]/mag[nz,None]
    t = np.radians(mag)[:,None]
    P = np.cos(t)*P0 + np.sin(t)*u
    P /= np.linalg.norm(P,axis=1,keepdims=True)
    P[is_vertex] = P0[is_vertex]
    return P, v

def W(kind, x, q):
    if kind == 'linear':     lam, = q; return np.clip(1-x/lam, 0, 1)
    if kind == 'power':      lam,p = q; return np.clip(1-x/lam, 0, 1)**p
    if kind == 'exp':        lam, = q; return np.exp(-x/lam)
    if kind == 'stretched':  lam,p = q; return np.exp(-(x/lam)**p)
    if kind == 'cos':        lam, = q; return np.cos((math.pi/2)*np.clip(x/lam,0,1))
    if kind == 'linear_dev': lam, = q; return np.clip(x/lam, 0, 1)
    if kind == 'power_dev':  lam,p = q; return np.clip(x/lam, 0, 1)**p
    raise ValueError(kind)

KINDS = ['linear','power','exp','stretched','cos','linear_dev','power_dev']
STARTS = {
 'linear':     [[0.004,5],[0.004,10],[0.006,3]],
 'power':      [[0.004,8,1],[0.005,15,2],[0.004,5,0.5]],
 'exp':        [[0.004,3],[0.004,6],[0.005,2]],
 'stretched':  [[0.004,5,1.5],[0.004,8,2],[0.005,3,1]],
 'cos':        [[0.004,8],[0.004,15],[0.005,5]],
 'linear_dev': [[0.004,20],[0.004,37],[0.005,30]],
 'power_dev':  [[0.004,25,1],[0.004,37,2],[0.005,20,0.5]],
}

def objective(z, kind):
    q = np.exp(z); a = q[0]; shape = q[1:]
    x = dev if kind.endswith('_dev') else d_e
    w = W(kind, x, shape)
    P,_ = build_P(a,w)
    return rho_fast(P)

# ================= 7. 联合优化 =================
print("\n================ 联合优化 (a, 形状参数) ================")
results = {}
for kind in KINDS:
    best = None
    for s0 in STARTS[kind]:
        z0 = np.log(np.array(s0,float))
        simp = np.array([z0]+[z0 + np.eye(len(z0))[i]*0.06 for i in range(len(z0))])
        res = minimize(objective, z0, args=(kind,), method='Nelder-Mead',
                       options={'maxiter':420,'initial_simplex':simp,
                                'xatol':1e-6,'fatol':1e-8})
        if best is None or res.fun < best[0]:
            best = (float(res.fun), np.exp(res.x).tolist(), int(res.nfev))
    q = best[1]; a = q[0]; shape = q[1:]
    x = dev if kind.endswith('_dev') else d_e
    w = W(kind, x, shape)
    P,v = build_P(a,w)
    results[kind] = dict(rho=best[0], a=a, shape=shape, nfev=best[2],
                         xvar='dev' if kind.endswith('_dev') else 'd_e',
                         w_at_edge=float(w[np.argmin(np.abs(d_e))]) if not kind.endswith('_dev') else None)
    print(f"  {kind:12s} rho={best[0]:.6f}deg  a={a:.7f}  shape={np.round(shape,4)}  nfev={best[2]}")
    json.dump(results, open(OUT+"/optimize_theta_joint_progress.json","w"), indent=2)

# ================= 8. 最优解：完整 Reynolds 复核 + 门禁 =================
best_kind = min(results, key=lambda k: results[k]['rho'])
q = [results[best_kind]['a']] + results[best_kind]['shape']
x = dev if best_kind.endswith('_dev') else d_e
w = W(best_kind, x, q[1:])
th = (math.pi/2)*np.clip(w,0,1)
d0 = (q[0]*dev*np.cos(th))[:,None]*u_rad + (q[0]*dev*np.sin(th))[:,None]*u_tan
v_full = reynolds_full(d0)
P_fast,_ = build_P(q[0], w)
P_full,_ = build_P(q[0], w, v_override=v_full)
r_fast, r_full = rho_fast(P_fast), rho_fast(P_full)
print(f"\n[复核] 最优族={best_kind}  快速分解 rho={r_fast:.9f}  完整 Reynolds rho={r_full:.9f}  "
      f"点集最大差={np.abs(P_fast-P_full).max():.3e}")
assert abs(r_fast-r_full) < 1e-8, "快速分解与完整 Reynolds 不一致"
P = P_full; r = r_full

def strict_sym(P,tol=1e-9):
    t = cKDTree(P); bad=0; worst=0.0
    for g in G120:
        d,j = t.query((g@P.T).T)
        if len(np.unique(j))!=len(P) or d.max()>tol: bad+=1
        worst=max(worst,float(d.max()))
    return bad, math.degrees(worst)

def mirror_median(P):
    A,B,Cc = V[0],V[4],V[8]; G=normalize(A+B+Cc)
    nA=normalize(np.cross(G,A)); S=np.eye(3)-2*np.outer(nA,nA)
    sel=np.where(fi==0)[0]; base=P0[sel]
    jj=cKDTree(base).query((S@base.T).T)[1]
    def td(Z):
        d=Z[sel]-base; return d-np.sum(d*base,axis=1)[:,None]*base
    d=td(P); Sd=(S@d.T).T
    num=np.sum(Sd*d[jj],axis=1)
    den=np.linalg.norm(Sd,axis=1)*np.linalg.norm(d[jj],axis=1)+1e-30
    return float(np.median(np.degrees(np.arccos(np.clip(num/den,-1,1))))), d, jj, Sd

def edge_checks(P):
    """三条棱：位移沿棱(符号均值/绝对值均值)、径向分量均值、与棱方向夹角"""
    sel=np.where(fi==0)[0]; base=P0[sel]
    d=P[sel]-base; d=d-np.sum(d*base,axis=1)[:,None]*base
    A,B,Cc=V[0],V[4],V[8]
    out=[]
    for cond,u in [(lambda i,j:j==0,normalize(B-A)),
                   (lambda i,j:j==i,normalize(Cc-A)),
                   (lambda i,j:i==n,normalize(Cc-B))]:
        ks=[h for h in range(len(sel)) if cond(int(I[sel][h]),int(J[sel][h]))]
        pr=np.array([np.dot(d[h],u) for h in ks])
        rad=np.array([np.dot(d[h],u_rad[sel][h]) for h in ks])
        cosang=np.array([np.dot(d[h],u)/(np.linalg.norm(d[h])+1e-30) for h in ks])
        out.append(dict(signed_mean=float(pr.mean()),abs_mean=float(np.abs(pr).mean()),
                        radial_mean=float(np.abs(rad).mean()),
                        angle_to_edge_deg=float(np.degrees(np.arccos(np.clip(np.abs(cosang),-1,1))).mean())))
    return out

bad, worst = strict_sym(P)
mir = mirror_median(P)[0]
ec = edge_checks(P)
print(f"\n[门禁] 点数={len(P)}  rho_max={r:.6f}deg  对称失败={bad}/120 偏差={worst:.9f}deg  "
      f"镜像median={mir:.6f}deg")
for i,e in enumerate(ec):
    print(f"       棱{i}: 沿棱符号均值={e['signed_mean']:.3e}  |沿棱|={e['abs_mean']:.6f}  "
          f"|径向|={e['radial_mean']:.6f}  与棱夹角={e['angle_to_edge_deg']:.3f}deg")
assert len(P)==7292
assert bad==0, f"严格对称失败 {bad}/120"
assert r<=1.5, f"rho_max={r} 超标"
assert mir<1e-3, f"镜像协变偏差 {mir}"
assert max(abs(e['signed_mean']) for e in ec)<1e-6, "存在统一环绕"
print("[门禁全部通过]")

# ================= 9. 交付 =================
np.save(OUT+"/v_theta_joint_best.npy", v_full)
coef = dict(variant="joint_theta_C", family=best_kind, xvar=results[best_kind]['xvar'],
            a=results[best_kind]['a'], shape=results[best_kind]['shape'],
            m_formula="m(dev)=a*dev", theta_formula="theta=(pi/2)*W(x), x=d_e or dev",
            W_forms=dict(linear="clip(1-d/lam,0,1)",power="clip(1-d/lam,0,1)^p",
                         exp="exp(-d/lam)",stretched="exp(-(d/lam)^p)",
                         cos="cos((pi/2)*clip(d/lam,0,1))",
                         linear_dev="clip(dev/lam,0,1)",power_dev="clip(dev/lam,0,1)^p"),
            rho_max=r, strict_sym_bad=bad, sym_worst_deg=worst,
            mirror_median_deg=mir, edge_checks=ec,
            baseline=dict(P0=ic.rho(P0), C_covariant_tangent=1.496089025393318,
                          A_pure_radial=1.492573), n_points=len(P))
json.dump(coef, open(OUT+"/theta_joint_coef_n27.json","w"), indent=2)
with open(OUT+"/theta_joint_points_n27.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j",
                                   "dev_deg","d_e_deg","theta_deg","disp_deg","is_vertex"])
    disp=[math.degrees(math.acos(np.clip(np.dot(P0[k],P[k]),-1,1))) for k in range(N)]
    for k in range(N):
        wr.writerow([P0[k,0],P0[k,1],P0[k,2],P[k,0],P[k,1],P[k,2],
                     int(fi[k]),int(I[k]),int(J[k]),dev[k],d_e[k],
                     math.degrees(th[k]),disp[k],int(is_vertex[k])])
json.dump(dict(results=results,best_kind=best_kind,rho=r,gate=dict(bad=bad,worst_deg=worst,
          mirror=mir,edges=ec),selfcheck=dict(inv_dev=inv_dev,inv_de=inv_de,
          decomp_err=err_dec,rho_ref=r_ref,rho_vec=r_new)),
          open(OUT+"/optimize_theta_joint.json","w"), indent=2)

# ================= 10. 出图 =================
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (a) W 曲线
    fig, ax = plt.subplots(1,2,figsize=(12,4.4))
    xd = np.linspace(0, float(d_e.max()), 200)
    for kind in KINDS:
        if kind.endswith('_dev'): continue
        q = results[kind]['shape']
        ax[0].plot(xd, W(kind,xd,q), lw=1.8, label=f"{kind} (rho={results[kind]['rho']:.5f})")
    ax[0].axvline(0,color='k',lw=.6); ax[0].set_xlabel('d_e: angular distance to nearest edge (deg)')
    ax[0].set_ylabel('w = theta / 90deg'); ax[0].set_title('Optimized deflection weight W(d_e)')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    xv = np.linspace(0,float(dev.max()),200)
    for kind in ['linear_dev','power_dev']:
        ax[1].plot(xv, W(kind,xv,results[kind]['shape']), lw=1.8,
                   label=f"{kind} (rho={results[kind]['rho']:.5f})")
    ax[1].set_xlabel('dev: angular distance to face center (deg)')
    ax[1].set_ylabel('w'); ax[1].set_title('Control: W(dev) (wrong independent variable)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT+"/fig_theta_joint_functions.png",dpi=140)

    # (b) rho 对比
    fig2, ax2 = plt.subplots(figsize=(9,4.2))
    names = ['P0','A radial','C origin'] + KINDS
    vals = [ic.rho(P0), 1.492573, 1.496089] + [results[k]['rho'] for k in KINDS]
    colors = ['gray','lightgray','steelblue'] + (['tab:orange']*5 + ['tab:red']*2)
    ax2.bar(range(len(names)), vals, color=colors, alpha=.85)
    ax2.axhline(1.5,color='red',ls='--',lw=1,label='1.5 deg limit')
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names,rotation=35,ha='right',fontsize=9)
    ax2.set_ylabel('rho_max (deg)'); ax2.set_title('Joint optimization: rho_max by deflection family')
    ax2.legend(); ax2.grid(axis='y',alpha=.3)
    for i,v0 in enumerate(vals): ax2.text(i,v0,f"{v0:.5f}",ha='center',va='bottom',fontsize=7)
    fig2.tight_layout(); fig2.savefig(OUT+"/fig_theta_joint_rho.png",dpi=140)
    print("[出图] fig_theta_joint_functions.png / fig_theta_joint_rho.png")
except Exception as e:
    print(f"[出图跳过] {type(e).__name__}: {e}")

print(f"\n================ 汇总 ================")
print(f"基线 P0={ic.rho(P0):.6f}  方案C=1.496089  方案A=1.492573")
for kind in KINDS:
    print(f"  {kind:12s} rho={results[kind]['rho']:.6f}  a={results[kind]['a']:.7f}  "
          f"shape={np.round(results[kind]['shape'],4)}")
print(f"  -> 最优 {best_kind}: rho={r:.6f}  "
      f"相对方案C改善={1.496089025393318-r:.6f}deg")
print(f"[交付] theta_joint_coef_n27.json / theta_joint_points_n27.csv / v_theta_joint_best.npy")
print(f"[完成] 用时 {time.time()-t_start:.1f}s")
