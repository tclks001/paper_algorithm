"""
球面覆盖半径 · 权威实现（2026-11-15 修正）
=========================================
⚠️ 本模块取代 improve_corrected.py 中 2016 起沿用的错误公式：

    2*atan( sqrt( sin(s-a)sin(s-b)sin(s-c)/sin s ) )

该式等于 **2r（球面内切圆半径的两倍）**，不是外接圆半径 R。
由欧拉型不等式 R >= 2r（等边取等）知它系统性低估覆盖半径。
详见 verify_circumradius_formula.py（向量几何 / Todhunter / 稠密采样三重复核）。

正确定义（向量几何，与 Song[29]、Larsen–Schmidt[32] 的"最大 Delaunay 外接圆半径"一致）：

    n = normalize( (b-a) × (c-a) )       # 过 a,b,c 的小圆圆心方向
    R = arccos( |n · a| )                # 球面外接圆半径（弧度）

等价的 Todhunter 球面包围圆公式（供交叉验证）：

    tan R = 2 sin(a/2) sin(b/2) sin(c/2)
            / sqrt( sin s · sin(s-a) sin(s-b) sin(s-c) ),   s=(a+b+c)/2

覆盖半径 rho(X) = max over Delaunay simplices of R（度）。
所有新脚本一律 import 本模块，不再各自内联半径公式。
"""
import numpy as np
import math
from scipy.spatial import ConvexHull


def sph_circumradius(a, b, c):
    """单个球面三角形的外接圆半径（弧度）。"""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    if ln < 1e-15:
        return math.pi
    n /= ln
    d = abs(float(np.dot(n, a)))
    return math.acos(max(-1.0, min(1.0, d)))


def circumradii(P):
    """返回点集球面 Delaunay（= 3D 凸包）所有三角形的外接圆半径（弧度）。"""
    P = np.asarray(P, float)
    t = ConvexHull(P).simplices
    a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ln = np.where(ln < 1e-15, 1.0, ln)
    n = n / ln[:, None]
    d = np.abs(np.sum(n * a, axis=1))
    return np.arccos(np.clip(d, -1.0, 1.0))


def circumradii_deg(P):
    """同上，单位：度。"""
    return np.degrees(circumradii(P))


def rho(P):
    """覆盖半径 = 最大 Delaunay 外接圆半径（度）。"""
    return float(np.degrees(circumradii(P).max()))


def todhunter(a, b, c):
    """Todhunter 球面外接圆公式（弧度），仅作交叉验证用。"""
    def ang(u, v):
        return math.acos(max(-1.0, min(1.0, float(np.dot(u, v)) /
                                      (np.linalg.norm(u) * np.linalg.norm(v)))))
    A, B, C = ang(a, b), ang(b, c), ang(c, a)
    s = (A + B + C) / 2
    val = math.sin(s) * math.sin(s - A) * math.sin(s - B) * math.sin(s - C)
    if val <= 0:
        return math.pi
    num = 2 * math.sin(A / 2) * math.sin(B / 2) * math.sin(C / 2)
    return math.atan2(num, math.sqrt(val))


if __name__ == "__main__":
    # 自检：向量定义 == Todhunter；等边时 CODE(旧公式) == R
    rng = np.random.default_rng(20261115)
    for _ in range(5):
        p = rng.normal(size=(3, 3)); p /= np.linalg.norm(p, axis=1)[:, None]
        r1 = sph_circumradius(*p); r2 = todhunter(*p)
        assert abs(r1 - r2) < 1e-9, (r1, r2)
    print("[自检通过] 向量几何定义 == Todhunter 球面外接圆公式")
