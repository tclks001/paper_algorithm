import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 正二十面体常数预备
# ---------------------------
phi = (1 + np.sqrt(5)) / 2  # 黄金分割率
# 正二十面体顶点
dots = [[0,1,phi],[0,1,-phi],[0,-1,phi],[0,-1,-phi],
        [1,phi,0],[1,-phi,0],[-1,phi,0],[-1,-phi,0],
        [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]]

def normalize(v):
    """
    归一化
    """
    return v / np.linalg.norm(v)

# 归一化
dots = [normalize(dot) for dot in dots]

# 正二十面体面
faces = [[0,4,8],[1,4,10],[2,5,8],[3,5,10],
         [0,6,9],[1,6,11],[2,7,9],[3,7,11],
         [0,2,8],[0,2,9],[1,3,10],[1,3,11],
         [4,6,0],[4,6,1],[5,7,2],[5,7,3],
         [8,10,4],[8,10,5],[9,11,6],[9,11,7]]


# ---------------------------
# 基础工具
# ---------------------------


def angle(a, b):
    """
    计算两个单位向量的夹角
    """
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

def slerp(a, b, t):
    """
    球面线性插值（SLERP）

    :param a: 单位向量1
    :param b: 单位向量2
    :param t: [0,1]内的插值参数
    """
    a = normalize(a)
    b = normalize(b)
    theta = angle(a, b)

    if theta < 1e-8:
        return a

    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return normalize(w1 * a + w2 * b)

# ---------------------------
# 边等角采样
# ---------------------------

def sample_edge(a, b, n):
    """
    在球面上对边 a---b 进行 n 段等角采样
    返回 n+1 个点
    """
    return [slerp(a, b, i / n) for i in range(n + 1)]


# ---------------------------
# 球面测地线（返回曲线上点集）
# ---------------------------

def geodesic_curve(a, b, steps=20):
    return [slerp(a, b, i / steps) for i in range(steps + 1)]

# ---------------------------
# 面内采样点生成
# ---------------------------

def spherical_face_sample(a, b, c, n):
    """
    在一个球面三角形面内生成近似等角采样点\n
    a b c 为表示三角形三个顶点的单位向量\n
    n 为每条边上的采样间隔数，即采样点数为 n+1
    """
    # 边采样（测地线的球面线性插值）
    ab = sample_edge(a, b, n)
    bc = sample_edge(b, c, n)
    ca = sample_edge(c, a, n)

    samples = []
    samples_map = {}
    temp_samples_1 = []
    temp_samples_2 = []
    lines = [] # 所有测地线，用于画图
    hots = [] # 描述两点间距离的热力图
    radius_hot = [] # 描述三角形外接圆半径的热力图
    # 在[ab,ca]坐标系中，这个点的两端分别为ab[i]和ca[n-i]
    # 这个点的序号为j，从0到i（长度为i+1）
    # 在[ca,bc]坐标系中，这个点的两端分别为ca[n-j]和bc[j]
    # 这个点的序号为i-j
    # 在[bc,ab]坐标系中，这个点的两端分别为bc[n-i+j]和ab[i-j]
    # 这个点的序号为n-i
    for i in range(0, n+1):
        for j in range(0, i+1):
            # 理论上这三个点会非常近
            sample1 = slerp(ab[i], ca[n-i], 0 if i==0 else j / i)
            sample2 = slerp(ca[n-j], bc[j], 0 if n-j==0 else (i-j) / (n-j))
            sample3 = slerp(bc[n-i+j], ab[i-j], 0 if n-i+j==0 else (n-i) / (n-i+j))
            samples.append(normalize(sample1 + sample2 + sample3))

            samples_map[(i,j,)] = samples[-1]
            #print(i,j, " -> ", samples[-1])
            
            
            temp_sample11 = slerp(bc[i], ab[n-i], 0 if i==0 else j / i)
            temp_sample12 = slerp(ab[n-j], ca[j], 0 if n-j==0 else (i-j) / (n-j))
            temp_sample13 = slerp(ca[n-i+j], bc[i-j], 0 if n-i+j==0 else (n-i) / (n-i+j))
            temp_samples_1.append(normalize(temp_sample11 + temp_sample12 + temp_sample13))

            temp_sample21 = slerp(ca[i], bc[n-i], 0 if i==0 else j / i)
            temp_sample22 = slerp(bc[n-j], ab[j], 0 if n-j==0 else (i-j) / (n-j))
            temp_sample23 = slerp(ab[n-i+j], ca[i-j], 0 if n-i+j==0 else (n-i) / (n-i+j))
            temp_samples_2.append(normalize(temp_sample21 + temp_sample22 + temp_sample23))

            
            
            if j > 0:
                coord = list(normalize(list(map(lambda a, b: (a+b), samples[-1], samples[-2]))))
                coord.append(angle(samples[-1], samples[-2]))
                hots.append(coord)

                temp_coord1 = list(normalize(list(map(lambda a, b: (a+b), temp_samples_1[-1], temp_samples_1[-2]))))
                temp_coord1.append(angle(temp_samples_1[-1], temp_samples_1[-2]))
                hots.append(temp_coord1)

                temp_coord2 = list(normalize(list(map(lambda a, b: (a+b), temp_samples_2[-1], temp_samples_2[-2]))))
                temp_coord2.append(angle(temp_samples_2[-1], temp_samples_2[-2]))
                hots.append(temp_coord2)
            # print(sample1, sample2, sample3, " -> ", normalize(sample1 + sample2 + sample3))
    
    # 求radius_hot
    # 每个小三角形都可以用三个点表示.
    # 点的坐标可以用在[ab,ca]坐标系中的坐标代替
    # 点[i,j]表示slerp(ab[i], ca[n-i], 0 if i==0 else j / i)
    # 如果是朝上的小三角形，三个点分别为[i,j],[i+1,j],[i+1,j+1]
    # 如果是朝下的小三角形，三个点分别为[i,j],[i,j+1],[i+1,j+1]
    for i in range(0, n):
        for j in range(0, i+1):
            sample1 = samples_map[(i,j,)]
            sample2 = samples_map[(i+1,j,)]
            sample3 = samples_map[(i+1,j+1,)]
            a = angle(sample1, sample2)
            b = angle(sample2, sample3)
            c = angle(sample3, sample1)
            p = (a+b+c)/2
            r = float(np.sqrt(p*(p-a)*(p-b)*(p-c)))
            radius_hot.append(list(normalize(sample1 + sample2 + sample3)) + [r])
            #print(a,b,c,p,r)
    # 画图
    for i in range(0, n+1):
        lines.append(geodesic_curve(ab[i], ca[n-i], steps=(i+1)*20))
        lines.append(geodesic_curve(bc[i], ab[n-i], steps=(i+1)*20))
        lines.append(geodesic_curve(ca[i], bc[n-i], steps=(i+1)*20))
        
    return samples, lines, hots, radius_hot

# ---------------------------
# Demo
# ---------------------------

if __name__ == "__main__":
    a, b, c = dots[faces[0][0]], dots[faces[0][1]], dots[faces[0][2]]
    N = 1000
    point_on_ab = slerp(a, b, 2/3)
    point_on_ac = slerp(a, c, 2/3)
    angle_pp = angle(point_on_ab, point_on_ac)
    #print(angle_pp*180*3/2/np.pi/np.sqrt(3))
    mid = (a+b+c)/3
    dist = np.linalg.norm(mid)
    print(angle(a, b)*180/np.pi/np.sqrt(3)/dist)



    line_pp = geodesic_curve(point_on_ab, point_on_ac, steps=20)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect((1,1,1))
    point_x, point_y, point_z = np.array(dots).T
    ax.scatter(point_x, point_y, point_z, c='b', marker='o')
    ax.scatter(a[0], a[1], a[2], c='r', marker='o')
    ax.scatter(b[0], b[1], b[2], c='r', marker='o')
    ax.scatter(c[0], c[1], c[2], c='r', marker='o')
    line_pp_x, line_pp_y, line_pp_z = np.array(line_pp).T

    ax.plot(line_pp_x, line_pp_y, line_pp_z, c='r')

    ax.set_axis_off()
    plt.show()