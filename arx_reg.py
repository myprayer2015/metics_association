import warnings

import statistics_func as sts
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv

# arx模型的fitness_score下限
FITNESS_MIN = 30
'''
arx模型的置信度的下限和理想值，如果在学习次数大于UPDATE_TIMES后，
若置信度大于理想值，则可以认为模型是valid，可以不再学习
若置信度小于下限，则认为模型不可靠，弃用之
'''
CONFIDENCE_MIN = 0.3
CONFIDENCE_VALID = 0.99
UPDATE_TIMES = 13

# 残差的阈值
RESIDUAL_THREHOLD = 0.8

# 系统异常得分下限
SYS_FAULTY = 0.5

'''
选定t为train_len，即训练数据的最后一个时刻
然后对比多组[n, m, k]，找出最优模型
'''
def getModels(item_name, item_data, time):
    n, m, k = [5, 10, 15, 20, 25], [5, 10, 15, 20, 25], [0, 0, 0, 0, 0]
    tmp_data = np.array(item_data)
    train_len = int(len(tmp_data[1]) * 0.8)
    train_data = tmp_data[:, :train_len].tolist()
    test_data = tmp_data[:, train_len:].tolist()
    test_time = time[train_len:]
    tt = test_time[0]
    for i in range(len(test_time)):
        test_time[i] -= tt
    '''
    功能：用于选出最优的arx模型
    xa, yb用来存x和y的系数，即arx模型的参数
    1.计算出参数
    2.根据参数推出预测值
    3.计算fitness_score
    4.选择fitness_score最大的模型
    c_score存储每对度量关联模型的置信度，如果c_score[i][j][0] == -1，则表示该对度量中至少有一个全为0或者在后期学习过程中被淘汰
    '''
    ya = np.zeros((82, 82, 0)).tolist()
    xb = np.zeros((82, 82, 0)).tolist()
    abk = np.zeros((82, 82)).tolist()
    c_score = np.zeros((82, 82, 0)).tolist()
    all_zeros = []
    for i in range(81):
        all_zeros.append(True)
    for i in range(81):
        for j in range(81):
            # 如果i或者j全为0，则不进行关联，只需要在i == 0时判断一遍，并更新all_zeros即可
            if i > 0 and (all_zeros[i] or all_zeros[j]):
                c_score[i][j].append(-1)
                c_score[j][i].append(-1)
                continue
            if i == 0:
                flag1 = True
                flag2 = True
                for q in range(len(train_data[i])):
                    if train_data[i][q][0] != 0:
                        flag1 = False
                    if train_data[j][q][0] != 0:
                        flag2 = False
                    if not flag1 and not flag2:
                        break
                all_zeros[i] = flag1
                all_zeros[j] = flag2
                if flag1 or flag2:
                    c_score[i][j].append(-1)
                    c_score[j][i].append(-1)
                    continue

            maxf = -1000
            for l in range(len(n)):
                st = max(n[l], m[l]+k[l]+1)
                # N = (n[l]+m[l]+1)
                N = 500
                a, b = leastSquares(n[l], m[l], k[l], st, N, train_data[i], train_data[j])
                if len(a) == 0 or len(b) == 0:
                    continue
                y1 = predict_y(a, b, k[l], train_data[i], train_data[j], st, N)
                # y的长度是N = n+m+1
                f_score = sts.fitness_score(train_data[j][st:st+N], y1)
                if f_score > maxf:
                    maxf = f_score
                    ya[i][j] = a
                    xb[i][j] = b
                    abk[i][j] = k[l]
            if maxf == -1000 or maxf < 0:
                c_score[i][j].append(-1)
            else:
                c_score[i][j].append(1 if maxf > FITNESS_MIN else 0)

    '''
    功能：进行模型validation验证，通过对长度为interval的k个时间段学习，更新置信度，如果仍然低于CONFIDENCE_MIN，则弃用该模型
    对于每对度量
    1.通过之前构建的模型预测y1
    2.计算模型的fitness_score
    3.根据fitness_score计算confidenc_score
    4.如果times > UPDATE_TIMES，根据confidence_score的大小决定弃用模型或者继续更新
    '''
    print("validation")
    times = 15
    interval = 10
    for i in range(81):
        for j in range(81):
            if c_score[i][j][0] != -1:
                for m in range(1, times+1):
                    y1 = predict_y(ya[i][j], xb[i][j], abk[i][j], train_data[i], train_data[j], (m-1)*interval+1, interval)
                    f_score = sts.fitness_score(train_data[j][(m-1)*interval+1:m*interval+1], y1)
                    c_score[i][j].append(sts.confidence_score(c_score[i][j][m-1], m, 1 if f_score > FITNESS_MIN else 0))
                    if m > UPDATE_TIMES:
                        #弃用
                        if c_score[i][j][m] < CONFIDENCE_MIN:
                            c_score[i][j][0] = -1
                            break
    # print(invalid)

    '''
    功能：根据正确的模型预测test_data，根据残差的分布情况给出度量可能出错的概率，即残差大于阈值的
    '''
    pred_data = np.zeros((82, 82, 0)).tolist()
    dists = np.zeros((82, 82, 0)).tolist()
    fdists = np.zeros((82, 82, 0)).tolist()
    for i in range(81):
        for j in range(81):
            if c_score[i][j][0] != -1:
                pd = predict_y(
                    ya[i][j], xb[i][j], abk[i][j], item_data[i], item_data[j], train_len, len(item_data[i])-train_len)
                for pdata in pd:
                    pred_data[i][j].append(pdata)
                z = []
                fz = []
                for r in range(len(pd)):
                    d = abs(pd[r]-item_data[j][r+train_len])
                    td = 1 if d > RESIDUAL_THREHOLD else 0
                    z.append(d)
                    fz.append(td)
                dists[i][j] = z
                fdists[i][j] = fz
    return ya, xb, k, c_score, test_time, dists

'''
功能：用最小二乘法找到使得误差最小的近似解
参数：n,m,k,st,N,x,y
    n,m,k为arx模型的参数
    st是选定的时刻，为了保证数据有效，要保证st > max(n, m+k+1)
    N表示选取多个时间点来训练，N = n+m
    x,y是预测关联度的两组数据
返回值：arx的参数xa, xb
'''
def leastSquares(n, m, k, st, N, x, y):
    l = np.zeros((n+m+1, n+m+1))
    r = np.zeros((n+m+1, 1))
    for i in range(N):
        time = st+i
        f = np.zeros((n+m+1, 1)).tolist()
        cnt = 0
        for j in range(1, n+1):
            f[cnt][0] = 0-y[time-j][0]
            cnt += 1
        for j in range(m+1):
            f[cnt][0] = x[time-k-j][0]
            cnt += 1
        fa = np.array(f)
        l += fa.dot(fa.T)
        r += fa*y[time][0]
    if np.linalg.det(l) != 0:
        ab = np.linalg.inv(l).dot(r)
        ya = ab[:n]
        xb = ab[n:]
    else:
        ya = xb = []
    return ya, xb

'''
功能：通过arx模型预测y
'''
def predict_y(ya, xb, k, x, y, st, N):
    y1 = []
    for i in range(N):
        time = st+i
        tsum = 0
        for j in range(1, len(ya)+1):
            tsum += ya[j-1]*y[time-j][0]
        for j in range(len(xb)):
            tsum += xb[j]*x[time-j-k][0]
        y1.append(-tsum)
    return y1

def drawResidual(test_time, c_score, item_name, dists, to_file):
    for i in range(81):
        for j in range(81):
            if c_score[i][j][0] != -1:
                x = test_time
                y = dists[i][j]
                up = [np.percentile(y, 75)]*len(y)
                plt.xlabel("time(ms)")
                plt.ylabel(str(item_name[j]))
                plt.title(str(item_name[i])+'---'+str(item_name[j]))
                plt.plot(x, y, c = 'b')
                plt.plot(x, up, c = 'r', ls = ':')
                plt.savefig(to_file+'//metrics//'+str(item_name[i])+'-'+str(item_name[j])+'.png')
                plt.close()
    return

def drawNormal(c_score, to_file):
    edgelist = []
    for i in range(81):
        for j in range(81):
            if c_score[i][j][0] != -1:
                edgelist.append((i, j))
    graph = nx.empty_graph(81)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist=[i for i in range(81)],
                           node_color='b', node_size=20, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=1, alpha=0.5, edge_color='black')
    plt.savefig(to_file + '//graph//random_layout.png')
    plt.close()
    return

'''
功能：用来求得某个时刻所有组件的异常得分
参数：times指模型优化迭代次数
'''
def judgeComponent(dists, c_score, times, test_time,  t, item_name, to_file):
    # val[i][0]表示与组件i相关的模型总数，val[i][1]表示报异常的模型总数
    val = np.zeros((81, 3)).tolist()
    for i in range(81):
        for j in range(81):
            if c_score[j][i][0] == -1:
                continue
            val[i][0] += 1
            if dists[j][i][t] > np.percentile(dists[j][i], 75):
                val[i][1] += 1
        if val[i][0] > 0:
            val[i][2] = 1.0*val[i][1]/val[i][0]
    with open(to_file+'//anomaly_component-'+ str(test_time[t])+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time: ' + str(test_time[t])])
        writer.writerow(['component', 'anomaly_score'])
        for i in range(len(val)):
            writer.writerow([item_name[i], val[i][2]])
    return val

'''
功能：求出某个时刻系统的异常得分，并根据阈值判定是否异常
参数：times指模型优化迭代次数
'''
def judgeSystem(dists, c_score, times, t):
    sum = 0
    cnt = 0
    for i in range(81):
        for j in range(81):
            if c_score[i][j][0] == -1:
                continue
            cnt += 1
            sum += c_score[i][j][times]*(dists[i][j][t])
    return True if (sum*1.0/cnt) > SYS_FAULTY else False

warnings.filterwarnings('ignore')
item_name, item_data, time = sts.loadData('data.csv')
ya, xb, k, c_score, test_time, dists = getModels(item_name, item_data, time)
# drawNormal(c_score, 'arx')
# drawResidual(test_time, c_score, item_name, dists, 'arx')
val = judgeComponent(dists, c_score, 15, test_time, 3, item_name, 'arx')