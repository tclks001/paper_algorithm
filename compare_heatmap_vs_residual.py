"""
对比：三角形外接圆半径热力图（用户 hot_radius.png 口径） vs 顶点 1-ring 平衡残差（思路 C）
================================================================================
热力图：标量场定义在【单纯形/小三角形】上，值 = 该三角形外接圆半径 phi(t)。
        回答"哪里最稀疏/最差"（全局绝对值分布）。
1-ring 残差：定义在【顶点】上，值 = 该点邻域内所有 phi(t) 的不均衡度
        r_i = max(phi in 1-ring) - min(phi in 1-ring)。
        回答"这个点还有没有可改进的局部移动"（局部一阶最优性）。
最优覆盖的等张力状态 = 所有 1-ring 内 phi 相等 => r_i -> 0。
"""
import numpy as np, math, json
from scipy.spatial import ConvexHull
import improve_corrected as ic

def normalize(v):
    v=np.asarray(v,float); s=np.linalg.norm(v); return v/s if s>1e-15 else v*0.0

n=27
P0,fi,I,J,isedge=ic.gen_meta(n)
Cs=np.array([normalize(ic.V[f[0]]+ic.V[f[1]]+ic.V[f[2]]) for f in ic.faces])

def build_sym(scale=1.0):
    v=np.load("E:/毕业论文/算法/v120_symmetric.npy")
    P=np.empty_like(P0)
    for k in range(len(P0)):
        if isedge[k] and np.linalg.norm(v[k])<1e-12: P[k]=P0[k]; continue
        if np.linalg.norm(v[k])<1e-12: P[k]=P0[k]; continue
        P[k]=ic.exp_map(P0[k],scale*np.linalg.norm(v[k]),normalize(v[k]))
    return P

def analyze(P,tag):
    h=ConvexHull(P)
    tris=h.simplices
    phi=np.array([math.degrees(ic.sph_circumradius(P[t[0]],P[t[1]],P[t[2]])) for t in tris])
    imax=int(np.argmax(phi))
    # 1-ring：把每个三角形的 phi 归属到它的 3 个顶点
    ring=[[] for _ in range(len(P))]
    for t,r in zip(tris,phi):
        for k in t: ring[k].append(r)
    rng=np.array([max(a)-min(a) for a in ring])
    std=np.array([np.std(a) for a in ring])
    # 最坏三角形重心到最近面心角距（对应热力图"红色在哪"）
    c=normalize(P[tris[imax]].sum(axis=0))
    dev=math.degrees(max(np.dot(c,x) for x in Cs))
    dev=math.degrees(math.acos(np.clip(max(np.dot(c,x) for x in Cs),-1,1)))
    worst_v=tris[imax]
    print(f"\n=== {tag} ===")
    print(f"  Delaunay 三角形数={len(tris)}  rho_max={phi.max():.6f}deg")
    print(f"  最坏三角形 #{imax}: phi={phi[imax]:.6f}deg, 重心到最近面心 dev={dev:.3f}deg")
    print(f"  最坏三角形 3 个顶点的 1-ring 极差 = {np.round(rng[worst_v],6)} deg")
    print(f"  全体顶点 1-ring 极差: min={rng.min():.6f} median={np.median(rng):.6f} max={rng.max():.6f} deg")
    print(f"  全体顶点 1-ring std : min={std.min():.6f} median={np.median(std):.6f} max={std.max():.6f} deg")
    print(f"  极差 <0.001deg 的顶点数 = {(rng<0.001).sum()} / {len(P)}  (局部已平衡)")
    return dict(tag=tag,rho=float(phi.max()),dev=float(dev),
                worst_ring=[float(x) for x in rng[worst_v]],
                rng_min=float(rng.min()),rng_med=float(np.median(rng)),rng_max=float(rng.max()),
                std_med=float(np.median(std)),balanced=int((rng<0.001).sum()))

r1=analyze(P0,"P0 原始 slerp 构造（热力图口径）")
r2=analyze(build_sym(1.5),"对称化修正场 s=1.5")
print("\n=== 结论判据 ===")
print(f"  rho_max: {r1['rho']:.6f} -> {r2['rho']:.6f}  (改善 {r1['rho']-r2['rho']:.6f}deg)")
print(f"  1-ring 极差中位数: {r1['rng_med']:.6f} -> {r2['rng_med']:.6f}")
print(f"  局部已平衡点数: {r1['balanced']} -> {r2['balanced']}")
json.dump([r1,r2],open("E:/毕业论文/算法/compare_heatmap_vs_residual.json","w"),indent=2)
print("\n[完成] compare_heatmap_vs_residual.json")
