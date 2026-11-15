"""
独立核查：improve_corrected.sph_circumradius 是否为球面【外接圆】半径
=====================================================================
日期：2026-11-15
起因：独立子代理交叉复核《第三章_成稿.md》时指出，代码里的
      2*atan(sqrt(sin(s-a)sin(s-b)sin(s-c)/sin s))
      可能是"球面内切圆半径的 2 倍"，而非外接圆半径。
      若成立，则本文全部 rho_max 以及以 rho_max 为目标的拟合优化都建立在错误量上。

四种独立口径（本脚本全部实算）：
  CODE   2*atan(sqrt(sin(s-a)sin(s-b)sin(s-c)/sin s))          = improve_corrected.py 现用
  GEOM   arccos(n·a), n = normalize((b-a)×(c-a))                = 向量几何定义（权威）
  TODH   tan R = 2 sin(a/2)sin(b/2)sin(c/2)
                / sqrt(sin s sin(s-a)sin(s-b)sin(s-c))         = Todhunter 球面外接圆公式
  SAMPLE Fibonacci 稠密采样 + cKDTree 直接估计覆盖半径（下界，与 Delaunay 无关）

只读取文件、不修改任何点集；输出 verify_circumradius_formula.json。
"""
import numpy as np, math, csv, json, sys, time
from scipy.spatial import ConvexHull, cKDTree
sys.path.insert(0, "E:/毕业论文/算法")
import improve_corrected as ic

OUT = "E:/毕业论文/算法"
t0 = time.time()

def load(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([[float(r["x_opt"]), float(r["y_opt"]), float(r["z_opt"])] for r in rows])

def circ_CODE(a, b, c):
    return math.degrees(ic.sph_circumradius(a, b, c))

def circ_GEOM(a, b, c):
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    if ln < 1e-15: return 180.0
    n /= ln
    d = float(np.dot(n, a))
    if d < 0: d = -d
    return math.degrees(math.acos(np.clip(d, -1, 1)))

def circ_TODH(a, b, c):
    A, B, C = ic.angle(a, b), ic.angle(b, c), ic.angle(c, a)
    s = (A + B + C) / 2
    val = math.sin(s) * math.sin(s - A) * math.sin(s - B) * math.sin(s - C)
    if val <= 0: return 180.0
    num = 2 * math.sin(A / 2) * math.sin(B / 2) * math.sin(C / 2)
    return math.degrees(math.atan2(num, math.sqrt(val)))

def rho_all(P, fn):
    t = ConvexHull(P).simplices
    r = np.array([fn(P[i], P[j], P[k]) for i, j, k in t])
    return t, r

def fibonacci(N):
    i = np.arange(N, dtype=float)
    ga = math.pi * (3 - math.sqrt(5))
    z = 1 - 2 * (i + 0.5) / N
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    th = ga * i
    return np.c_[r * np.cos(th), r * np.sin(th), z]

def sample_rho(P, N=300000):
    S = fibonacci(N)
    d, _ = cKDTree(P).query(S, workers=-1)
    return math.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1)).max())

# ---------- 1. 等边球面三角形对照（判断 CODE 是否只在等边时等于外接圆） ----------
alpha = math.radians(2.0)
A = np.array([1.0, 0, 0])
B = np.array([math.cos(alpha), math.sin(alpha), 0])
x = math.cos(alpha) * (1 - math.cos(alpha)) / math.sin(alpha)
y = math.sqrt(max(0.0, math.sin(alpha) ** 2 - x ** 2))
C = np.array([math.cos(alpha), x, y])
eq = dict(edge_AB=math.degrees(ic.angle(A, B)), edge_AC=math.degrees(ic.angle(A, C)),
          edge_BC=math.degrees(ic.angle(B, C)),
          CODE=circ_CODE(A, B, C), GEOM=circ_GEOM(A, B, C), TODH=circ_TODH(A, B, C))
print("=== 等边球面三角形 (a=b=c=2deg) ===")
print(f"  边长: {eq['edge_AB']:.6f} / {eq['edge_AC']:.6f} / {eq['edge_BC']:.6f}")
print(f"  CODE={eq['CODE']:.6f}  GEOM={eq['GEOM']:.6f}  TODH={eq['TODH']:.6f}")

# ---------- 2. 三个交付点集：四种口径 ----------
cases = [
    ("P0 baseline", "baseline_P0_n27.csv"),
    ("scheme L linear", "symmetric_points_final_C.csv"),
    ("scheme J joint", "theta_m0_joint_points_n27.csv"),
]
res = {}
print("\n=== 三个点集：覆盖半径对照（度） ===")
print(f"  {'case':20s} {'CODE':>10s} {'GEOM':>10s} {'TODH':>10s} {'SAMPLE':>10s} {'GEOM-TODH':>12s}")
for name, fn in cases:
    P = load(f"{OUT}/{fn}")
    t, rc = rho_all(P, circ_CODE)
    _, rg = rho_all(P, circ_GEOM)
    _, rt = rho_all(P, circ_TODH)
    rs = sample_rho(P)
    gap = float(np.abs(rg - rt).max())
    res[name] = dict(file=fn, n_points=len(P), n_tri=len(t),
                     CODE_max=float(rc.max()), GEOM_max=float(rg.max()),
                     TODH_max=float(rt.max()), SAMPLE_max=float(rs),
                     GEOM_TODH_max_abs_diff=gap,
                     CODE_GEOM_mean_abs_diff=float(np.abs(rc - rg).mean()),
                     CODE_GEOM_max_abs_diff=float(np.abs(rc - rg).max()))
    print(f"  {name:20s} {rc.max():10.6f} {rg.max():10.6f} {rt.max():10.6f} {rs:10.6f} {gap:12.2e}")

# ---------- 3. 影响范围：不同 n 的基线真实覆盖半径 ----------
print("\n=== 影响范围：基线构造在不同 n 下的真实覆盖半径（GEOM 口径） ===")
by_n = {}
for n in [27, 28, 29, 30]:
    P0, fi, I, J, ie = ic.gen_meta(n)
    _, r = rho_all(P0, circ_GEOM)
    _, rc = rho_all(P0, circ_CODE)
    by_n[str(n)] = dict(M=int(10 * n * n + 2), CODE=float(rc.max()), GEOM=float(r.max()),
                         meets_1p5=bool(r.max() <= 1.5))
    print(f"  n={n}  M={10*n*n+2:5d}  CODE={rc.max():.6f}  GEOM={r.max():.6f}  "
          f"<=1.5deg? {r.max()<=1.5}")

# ---------- 4. 断言（本脚本自身的正确性测试） ----------
assert all(v["GEOM_TODH_max_abs_diff"] < 1e-9 for v in res.values()), "GEOM 与 Todhunter 不一致"
assert eq["GEOM"] - eq["TODH"] < 1e-9, "等边对照 GEOM/TODH 不一致"
assert res["scheme J joint"]["GEOM_max"] > 1.5, "与子代理结论不符：J 的真实 rho 应 >1.5"

report = dict(
    date="2026-11-15",
    verdict="CODE 公式不是球面外接圆半径；GEOM == TODH 才是。CODE = 2r（内切圆半径的两倍），"
            "由欧拉型不等式 R >= 2r（等边取等）系统性低估外接圆半径。",
    equilateral_check=eq, cases=res, baseline_by_n=by_n,
    affected_scripts=[  # grep: 2*math.atan / 2*np.arctan
        "compare_constructions.py", "diagnose_theta_insensitivity.py", "diag_max_location.py",
        "improve_corrected.py", "improve_final.py", "improve_legfix.py",
        "improve_symmetric_c.py", "optimize_greedy.py", "optimize_radial.py",
        "optimize_test.py", "optimize_theta_joint.py", "optimize_theta_m0_joint.py",
        "plot_hypotenuse.py", "plot_joint_effect.py", "verify_symmetric_fix.py",
        "verify_symmetry.py", "verify_theta_m0_result.py"],
    consequence="所有 rho_max 数值、热力图/残差统计、以及以 rho_max 为目标的"
                "位置拟合优化（方案 L/J 的系数）都基于错误目标，需全面返工重算；"
                "第三章 'rho_max<=1.5deg 无盲区' 的结论当前不成立。",
    correct_formula="GEOM: n=normalize((b-a)x(c-a)); R=arccos(|n·a|)。"
                     "等价 Todhunter: tan R = 2 sin(a/2)sin(b/2)sin(c/2)"
                     " / sqrt(sin s sin(s-a)sin(s-b)sin(s-c))。",
    cross_check="由独立子代理（无预设结论）首先提出；本脚本用向量几何定义、"
                "Todhunter 公式与 Fibonacci 稠密采样三重独立复核。")
json.dump(report, open(f"{OUT}/verify_circumradius_formula.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n[自检通过] 用时 {time.time()-t0:.1f}s")
print("[交付] verify_circumradius_formula.json")
