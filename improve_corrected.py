"""
改进构造 · 边界感知 + 结构性单调修正（修复"棱上点被径向拉离导致裂缝"的 bug）
=========================================================================
修复点：
  1) 落在二十面体棱上的点（is_edge）改为沿棱切向移动，不再径向离开棱；
     保证 20 面拼接时共享边界仍吻合（无 crack）。
  2) 12 个二十面体顶点固定不动（拓扑锚点）。
  3) 幅度 m(dev)=u0² + u1²·dev + u2²·dev²（三项均 ≥0 ⇒ 对 dev 单调不减、平滑），
     结构性杜绝上版的振铃/负值非物理解。
  4) 小步长优化：极小初始单纯形做 Nelder-Mead 精修。

对称：保持 20 面 6 折对称（初值对称 ⇒ Voronoi 对称 ⇒ 修正后仍对称）。
交付：corrected_points_n27.csv、corrected_sextant_n27.csv、fig_m_dev_corrected.png、fig_movefield_corrected.png。
"""
import numpy as np, math, time, json, os
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import csv

phi=(1+5**0.5)/2
dots=np.array([[0,1,phi],[0,1,-phi],[0,-1,phi],[0,-1,-phi],
               [1,phi,0],[1,-phi,0],[-1,phi,0],[-1,-phi,0],
               [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]],float)
V=dots/math.sqrt(1+phi*phi)
faces=[[0,4,8],[1,4,10],[2,5,8],[3,5,10],
       [0,6,9],[1,6,11],[2,7,9],[3,7,11],
       [0,2,8],[0,2,9],[1,3,10],[1,3,11],
       [4,6,0],[4,6,1],[5,7,2],[5,7,3],
       [8,10,4],[8,10,5],[9,11,6],[9,11,7]]

def normalize(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>0 else v
def angle(a,b): return float(np.arccos(np.clip(np.dot(normalize(a),normalize(b)),-1,1)))
def slerp(a,b,t):
    a,b=normalize(a),normalize(b); th=angle(a,b)
    if th<1e-12: return a.copy()
    s=math.sin(th); return normalize(math.sin((1-t)*th)/s*a+math.sin(t*th)/s*b)

def gen_meta(n):
    pts=[]; fid=[]; Ii=[]; Jj=[]; isedge=[]
    for fi,f in enumerate(faces):
        a,b,c=V[f[0]],V[f[1]],V[f[2]]
        ab=[slerp(a,b,i/n) for i in range(n+1)]; bc=[slerp(b,c,i/n) for i in range(n+1)]; ca=[slerp(c,a,i/n) for i in range(n+1)]
        for i in range(n+1):
            for j in range(i+1):
                s1=slerp(ab[i],ca[n-i],0 if i==0 else j/i); s2=slerp(ca[n-j],bc[j],0 if n-j==0 else (i-j)/(n-j)); s3=slerp(bc[n-i+j],ab[i-j],0 if n-i+j==0 else (n-i)/(n-i+j))
                pts.append(normalize(s1+s2+s3)); fid.append(fi); Ii.append(i); Jj.append(j)
                isedge.append((i==0) or (j==0) or (j==i))
    seen={}; keep=[]
    for k,p in enumerate(pts):
        key=tuple(np.round(p,10))
        if key not in seen: seen[key]=k; keep.append(k)
    return (np.array([pts[k] for k in keep]), np.array(fid)[keep],
            np.array(Ii)[keep], np.array(Jj)[keep], np.array(isedge)[keep])

def sph_circumradius(a,b,c):
    """球面三角形【外接圆】半径（弧度）。
    ⚠️ 2026-11-15 修正：本函数原为 2*atan(sqrt(sin(s-a)sin(s-b)sin(s-c)/sin s))，
    那是 2r（球面内切圆半径的两倍）而非外接圆半径 R，会系统性低估覆盖半径。
    正确定义见 covering_radius.py 与 verify_circumradius_formula.py。
    """
    a,b,c=np.asarray(a,float),np.asarray(b,float),np.asarray(c,float)
    n=np.cross(b-a,c-a); ln=np.linalg.norm(n)
    if ln<1e-15: return math.pi
    n/=ln
    return math.acos(max(-1.0,min(1.0,abs(float(np.dot(n,a))))))

def rho(P):
    """覆盖半径 = 最大 Delaunay 外接圆半径（度），向量化正确实现。"""
    P=np.asarray(P,float); t=ConvexHull(P).simplices
    a,b,c=P[t[:,0]],P[t[:,1]],P[t[:,2]]
    n=np.cross(b-a,c-a); ln=np.linalg.norm(n,axis=1)
    n=n/np.where(ln<1e-15,1.0,ln)[:,None]
    d=np.abs(np.sum(n*a,axis=1))
    return math.degrees(float(np.arccos(np.clip(d,-1.0,1.0)).max()))
def fejes(M): w=M*math.pi/(6*M-12); return math.degrees(math.acos((1/math.sqrt(3))*math.cos(w)/math.sin(w)))

def exp_map(p, ang_deg, u):
    th=math.radians(ang_deg); return normalize(math.cos(th)*p + math.sin(th)*u)

def build_fields(P0, fi):
    Cs=np.array([normalize(V[faces[k][0]]+V[faces[k][1]]+V[faces[k][2]]) for k in range(20)])
    N=len(P0)
    C=np.array([Cs[fi[k]] for k in range(N)])
    dev=np.array([math.degrees(angle(P0[k],C[k])) for k in range(N)])
    is_vertex=np.zeros(N,bool)
    for v in V:
        d=np.array([angle(P0[k],v) for k in range(N)])
        is_vertex |= (d<1e-6)
    u_edge=np.zeros((N,3))
    for k in range(N):
        f=faces[fi[k]]; A,B,Cc=V[f[0]],V[f[1]],V[f[2]]
        edges=[(A,B),(B,Cc),(Cc,A)]
        best=None; best_err=1e9
        for (X,Y) in edges:
            err=abs(angle(P0[k],X)+angle(P0[k],Y)-angle(X,Y))
            if err<best_err: best_err=err; best=(X,Y)
        X,Y=best
        n_e=normalize(np.cross(X,Y))
        u_edge[k]=normalize(np.cross(P0[k],n_e))
    # 到棱的 barycentric 距离 min_bary（0=在棱上），用于平滑过渡
    min_bary=np.zeros(N)
    for k in range(N):
        f=faces[fi[k]]; A,B,Cc=V[f[0]],V[f[1]],V[f[2]]
        n_plane=normalize(np.cross(B-A,Cc-A))
        pp=P0[k]-np.dot(P0[k],n_plane)*n_plane
        v0=B-A; v1=Cc-A; v2=pp-A
        d00=np.dot(v0,v0); d01=np.dot(v0,v1); d11=np.dot(v1,v1); d20=np.dot(v2,v0); d21=np.dot(v2,v1)
        denom=d00*d11-d01*d01
        if abs(denom)>1e-20:
            al=(d11*d20-d01*d21)/denom; be=(d00*d21-d01*d20)/denom; ga=1-al-be
            min_bary[k]=max(0.0,min(al,be,ga))
    return C,dev,is_vertex,u_edge,min_bary,Cs

class Corrector:
    """边界感知 + 结构性单调幅度。
    内部点沿面心径向；棱上点沿棱切向；顶点不动。
    m(dev)=u0² + u1²·dev + u2²·dev²  ⇒ 对 dev 单调不减、平滑。"""
    def __init__(self, P0, fi, C, dev, is_vertex, u_edge, is_edge, min_bary, band=0.08):
        self.P0=P0; self.fi=fi; self.C=C; self.dev=dev
        self.is_vertex=is_vertex; self.u_edge=u_edge; self.is_edge=is_edge
        self.min_bary=min_bary; self.band=band
        self.N=len(P0)
        self.u_rad=np.array([normalize(C[k]-np.dot(C[k],P0[k])*P0[k]) for k in range(self.N)])
    def dir_of(self,k):
        # 平滑过渡：内部纯径向(w_b=0)，越靠近棱越切向(w_b→1)
        if self.is_vertex[k]:
            return self.u_rad[k]
        wb=min(1.0, max(0.0, (self.band - self.min_bary[k]) / self.band))
        if wb<=1e-6:
            return self.u_rad[k]
        if wb>=1-1e-6:
            return self.u_edge[k]
        u=normalize((1-wb)*self.u_rad[k] + wb*self.u_edge[k])
        return u
    def m_of(self, raw):
        u0,u1,u2=raw
        return u0*u0 + u1*u1*self.dev + u2*u2*self.dev**2
    def apply(self, raw):
        m=self.m_of(raw)
        P=np.empty_like(self.P0)
        for k in range(self.N):
            if self.is_vertex[k]:
                P[k]=self.P0[k]; continue
            if abs(m[k])<1e-9: P[k]=self.P0[k]; continue
            P[k]=exp_map(self.P0[k], m[k], self.dir_of(k))
        return P
    def objective(self, raw): return rho(self.apply(raw))
    def rho_of(self, raw): return rho(self.apply(raw))

def extract_field(P0,Popt,fi,Cs,face_select=0):
    sel=(fi==face_select)
    P0s=P0[sel]; Popts=Popt[sel]; C=Cs[face_select]
    ref=normalize(V[faces[face_select][0]]-np.dot(V[faces[face_select][0]],C)*C)
    perp=np.cross(C,ref)
    devs=[]; mags=[]; psis=[]; azs=[]
    for k in range(len(P0s)):
        p=P0s[k]; q=Popts[k]
        d=angle(p,C); u=normalize(p-np.dot(p,C)*C); az=math.degrees(math.atan2(np.dot(u,perp),np.dot(u,ref)))
        if az<0: az+=360
        if not(0<=az<60): continue
        cos_t=np.clip(np.dot(p,q),-1,1); t=np.arccos(cos_t)
        if t<1e-12: mags.append(0.0); psis.append(0.0)
        else:
            v=t*(q-cos_t*p)/math.sin(t); urad=normalize(C-np.dot(C,p)*p)
            cr=np.dot(v,urad); vt=v-cr*urad; ct=np.linalg.norm(vt)
            psi=0.0 if ct<1e-12 else (1 if np.dot(np.cross(urad,vt/ct),p)>=0 else -1)*math.degrees(math.atan2(ct,cr))
            mags.append(math.degrees(t)); psis.append(psi)
        devs.append(math.degrees(d)); azs.append(az)
    return dict(dev=np.array(devs), m=np.array(mags), psi=np.array(psis), az=np.array(azs), n=len(devs))

def main():
    n=27; M=10*n*n+2
    cache="E:/毕业论文/算法/corrected_coef_n27.json"
    P0,fi,I,J,isedge=gen_meta(n)
    C,dev,is_vertex,u_edge,min_bary,Cs=build_fields(P0,fi)
    base=rho(P0); lb=fejes(M)
    print(f"=== n={n} M={M} (边界感知 + 结构性单调 + 平滑过渡) ===")
    print(f"基线 ρ_max={base:.6f}°  下界={lb:.6f}°  gap={base-lb:.6f}°")
    print(f"顶点固定数={is_vertex.sum()}  棱上点数(含顶点)={isedge.sum()}")
    corr=Corrector(P0,fi,C,dev,is_vertex,u_edge,isedge,min_bary)
    print(f"[零修正] ρ_max={corr.rho_of([0,0,0]):.6f}°")
    if os.path.exists(cache):
        raw=json.load(open(cache))['raw']; print(f"[载入缓存] ρ_max={corr.rho_of(raw):.6f}° raw={np.round(raw,5)}")
    else:
        t0=time.time()
        res=minimize(lambda c: corr.objective(c), x0=[0,0,0], method='Nelder-Mead',
                     options={'maxiter':800,'initial_simplex':np.array([[0,0,0],[1e-3,0,0],[0,1e-3,0],[0,0,1e-3]]),
                              'xatol':1e-7,'fatol':1e-8})
        raw=list(res.x); json.dump({'raw':raw}, open(cache,'w'))
        print(f"[小步长优化] 最优 ρ_max={res.fun:.6f}°  用时{time.time()-t0:.1f}s")
    Popt=corr.apply(raw)
    rho_opt=rho(Popt)
    u0,u1,u2=raw; p,q,r=u0*u0,u1*u1,u2*u2
    print(f"[修正后] ρ_max={rho_opt:.6f}°  ≤1.5°? {'YES' if rho_opt<=1.5 else 'NO'}  改善 {base-rho_opt:.6f}°")
    print(f"  m(dev)={p:.6f} + {q:.6f}·dev + {r:.8f}·dev²   (p,q,r≥0 ⇒ 单调不减)")
    dd=np.linspace(0,dev.max(),9); mm=[p+q*x+r*x*x for x in dd]
    print("  m(dev) 采样: " + ", ".join(f"{d:.1f}°→{m:+.4f}°" for d,m in zip(dd,mm)))
    field=extract_field(P0,Popt,fi,Cs,0)
    print(f"[位移场] 1/6域点数={field['n']}  m∈[{field['m'].min():.4f},{field['m'].max():.4f}]°  |ψ|max={np.abs(field['psi']).max():.3f}°")
    cm=np.polyfit(field['dev'],field['m'],2)
    r2=1-np.sum((field['m']-np.polyval(cm,field['dev']))**2)/np.sum((field['m']-field['m'].mean())**2)
    print(f"[拟合] m(dev) R²={r2:.5f}  系数(高→低)={np.round(cm,6)}")
    with open("E:/毕业论文/算法/corrected_points_n27.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["x","y","z","x_opt","y_opt","z_opt","face_idx","i","j","dev_deg","m_deg","psi_deg","is_edge","is_vertex"])
        for k in range(len(P0)):
            q2=Popt[k]; cos_t=np.clip(np.dot(P0[k],q2),-1,1); t=np.arccos(cos_t)
            if t<1e-12: psi=0.0
            else:
                vv=t*(q2-cos_t*P0[k])/math.sin(t); ur=corr.u_rad[k]; cr=np.dot(vv,ur); vt2=vv-cr*ur; ctn=np.linalg.norm(vt2)
                psi=0.0 if ctn<1e-12 else (1 if np.dot(np.cross(ur,vt2/ctn),P0[k])>=0 else -1)*math.degrees(math.atan2(ctn,cr))
            w.writerow([P0[k,0],P0[k,1],P0[k,2],q2[0],q2[1],q2[2],int(fi[k]),int(I[k]),int(J[k]),dev[k],math.degrees(t) if t>=1e-12 else 0.0,psi, int(isedge[k]), int(is_vertex[k])])
    with open("E:/毕业论文/算法/corrected_sextant_n27.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["dev_deg","m_deg","psi_deg","az_deg"])
        for d,m,p,a in zip(field['dev'],field['m'],field['psi'],field['az']): w.writerow([d,m,p,a])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,4.5))
        xd=np.linspace(field['dev'].min(),field['dev'].max(),100)
        ax[0].scatter(field['dev'],field['m'],s=12,c='steelblue',alpha=.6,label='samples')
        ax[0].plot(xd,np.polyval(cm,xd),'r-',label=f'fit m(dev) R²={r2:.4f}')
        ax[0].set_xlabel('dev (°)'); ax[0].set_ylabel('m (°)'); ax[0].set_title('Function ① m(dev) [monotonic, boundary-aware]'); ax[0].legend()
        ax[1].hist(field['psi'],bins=30,color='darkgreen',alpha=.7); ax[1].set_xlabel('ψ (°)'); ax[1].set_ylabel('count'); ax[1].set_title('Function ② ψ(dev) (0=radial, ±90=tangential)')
        fig.tight_layout(); fig.savefig("E:/毕业论文/算法/fig_m_dev_corrected.png",dpi=130)
        fig2,ax2=plt.subplots(figsize=(6.5,5.5))
        ur=field['m']*np.cos(np.radians(field['psi'])); ut=field['m']*np.sin(np.radians(field['psi']))
        ax2.quiver(field['az'],field['dev'],ut,ur,field['m'],cmap='plasma',angles='xy',scale_units='xy',scale=1.0)
        ax2.set_xlabel('azimuth in sextant (°)'); ax2.set_ylabel('dev (°)'); ax2.set_title('Correction field on 1/6 domain\n(ψ→±90° at edges = tangential)')
        fig2.tight_layout(); fig2.savefig("E:/毕业论文/算法/fig_movefield_corrected.png",dpi=130)
        print("[出图] fig_m_dev_corrected.png, fig_movefield_corrected.png")
    except Exception as e: print(f"[出图跳过] {e}")
    print("\n================ 汇总 ================")
    print(f"n=27 M={M}: 基线 ρ_max={base:.6f}° → 修正后 ρ_max={rho_opt:.6f}°  (≤1.5°: {'YES' if rho_opt<=1.5 else 'NO'})")
    print(f"Fejes Tóth 下界={lb:.6f}°  修正后 gap={rho_opt-lb:.6f}° ({(rho_opt-lb)/lb*100:.3f}%)")
    print(f"对照：上版 radial-only 修正 ρ_max=1.496142°(非单调振铃)；n=28 基线=1.448773°")

if __name__=='__main__':
    main()
