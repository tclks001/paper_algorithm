"""
演示：为什么"三个边上的点都沿边移动"仍会不对称。
核心：取一条二十面体棱 AB（被两个面 f1、f2 共用），在 AB 上取一对 2 重旋转镜像点 p（靠近 A）与 p'=g(p)（靠近 B）。
这两个点到面心距离相同、到棱距离都是 0，但算法把 p 归属 f1、p' 归属 f2。
f1、f2 绕共享棱 AB 的缠绕方向相反 -> 算法算出的"沿棱切向"符号相反 -> 不对称。
"""
import numpy as np, math

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

def gen_meta(n):
    pts=[]; fid=[]; isedge=[]
    for fi,f in enumerate(faces):
        a,b,c=V[f[0]],V[f[1]],V[f[2]]
        ab=[normalize(a*math.sin((1-i/n)*angle(a,b))/math.sin(angle(a,b))+b*math.sin(i/n*angle(a,b))/math.sin(angle(a,b))) for i in range(n+1)]
        bc=[normalize(b*math.sin((1-i/n)*angle(b,c))/math.sin(angle(b,c))+c*math.sin(i/n*angle(b,c))/math.sin(angle(b,c))) for i in range(n+1)]
        ca=[normalize(c*math.sin((1-i/n)*angle(c,a))/math.sin(angle(c,a))+a*math.sin(i/n*angle(c,a))/math.sin(angle(c,a))) for i in range(n+1)]
        for i in range(n+1):
            for j in range(i+1):
                s1=ab[i]; s2=ca[n-i] if i==0 else normalize(s1*(1-j/i)+ca[n-i]*(j/i)); s3=bc[n-i+j] if n-i+j==0 else normalize(bc[n-i+j]*(1-(n-i)/(n-i+j))+ab[i-j]*((n-i)/(n-i+j)))
                pts.append(normalize(s1+s2+s3)); fid.append(fi); isedge.append((i==0) or (j==0) or (j==i))
    seen={}; keep=[]
    for k,p in enumerate(pts):
        key=tuple(np.round(p,10))
        if key not in seen: seen[key]=k; keep.append(k)
    return np.array([pts[k] for k in keep]), np.array(fid)[keep], np.array(isedge)[keep]

def rot_axis(k,theta):
    k=normalize(k); c=math.cos(theta); s=math.sin(theta)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(k,k)

def build_u_edge(P0,fi):
    N=len(P0); u_edge=np.zeros((N,3))
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
    return u_edge

def build_C(P0,fi):
    Cs=np.array([normalize(V[faces[k][0]]+V[faces[k][1]]+V[faces[k][2]]) for k in range(20)])
    return np.array([Cs[fi[k]] for k in range(len(P0))])

n=27
P0,fi,isedge=gen_meta(n)
u_edge=build_u_edge(P0,fi)
C=build_C(P0,fi)

# 取共享棱 AB：face0=[0,4,8] 与 face12=[4,6,0] 共用棱 (0,4)
A,B=V[0],V[4]
F=normalize(A+B)                      # 棱中点方向
g=rot_axis(F,math.pi)                # 绕"过棱中点的2重轴"转180度

# 找出落在棱 AB 上的点（err(AB) 极小）
errs=np.array([abs(angle(P0[k],A)+angle(P0[k],B)-angle(A,B)) for k in range(len(P0))])
onAB=np.where(errs<1e-6)[0]
print(f"落在棱 AB 上的点数 = {len(onAB)}")
# 选一个靠近 A 的点（到 A 的角距小）作为 p
order=onAB[np.argsort([angle(P0[k],A) for k in onAB])]
p_idx=order[1]                       # order[0] 通常是顶点 A 本身
p=P0[p_idx]
p2=g@p                               # 理论镜像点
# 在集合中找 p2 的最近点
idx2=int(np.argmin([angle(P0[k],p2) for k in range(len(P0))]))
q_idx=idx2; q=P0[q_idx]

print(f"\n=== 一对 2 重旋转镜像点 ===")
print(f"p (靠近 A): 索引={p_idx}  fi={fi[p_idx]}  dev(到面心)={math.degrees(angle(p,C[p_idx])):.4f}°")
print(f"q=g(p) (靠近 B): 索引={q_idx}  fi={fi[q_idx]}  dev(到面心)={math.degrees(angle(q,C[q_idx])):.4f}°")
print(f"  -> dev 是否相等: {abs(math.degrees(angle(p,C[p_idx]))-math.degrees(angle(q,C[q_idx])))<1e-6}")
print(f"\n=== 算法给的'沿棱切向'方向 u_edge ===")
print(f"u_edge[p] = {np.round(u_edge[p_idx],4)}")
print(f"u_edge[q] = {np.round(u_edge[q_idx],4)}")
print(f"g·u_edge[p] = {np.round(g@u_edge[p_idx],4)}")
cos1=np.dot(u_edge[q_idx], g@u_edge[p_idx])
print(f"cos(u_edge[q], g·u_edge[p]) = {cos1:.6f}   (理想对称应=+1；这里≈{-1 if cos1<-0.5 else cos1:.4f})")
cos0=np.dot(u_edge[p_idx],u_edge[q_idx])
print(f"cos(u_edge[p], u_edge[q])   = {cos0:.6f}   (≈ -1 说明两者被推往相反方向)")

# 结论性演示：若对称，则位移向量应满足 g·dir(p) = dir(q)。这里用的是 edge 点 -> dir=u_edge。
print(f"\n=== 同一物理棱 AB，但两个相邻面 f1、f2 的'最近棱'选取顺序 ===")
f1,f2=0,12
print(f"face {f1} 顶点顺序 = {faces[f1]}  -> 共享棱 (0,4) 在其 edges 列表里是 {[faces[f1][0],faces[f1][1]]} (正向)")
print(f"face {f2} 顶点顺序 = {faces[f2]}  -> 共享棱 (0,4) 在其 edges 列表里是 {[faces[f2][2],faces[f2][0]]} (反向!)")
print(f"  正向时 n_e=cross(A,B)，反向时 n_e=cross(B,A)=-cross(A,B) -> u_edge 符号相反")
