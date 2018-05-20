import warnings
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import networkx as nx
import statistics_func as sts

# 模型的最低分数
MIN_SCORE = 0.9
# 需要进行异常检测的时间点
detect_time = 260
# 数据平滑化的参数 K
K = 6

'''
功能：根据训练数据构建线性回归模型，返回每个模型的得分，预测结果等数据
'''
def linearReg(tmp_data, time):
    cf = np.zeros((81, 81)).tolist()
    it = np.zeros((81, 81)).tolist()
    sc = np.zeros((81, 81)).tolist()
    print(tmp_data[0])
    item_data = np.array(tmp_data)
    train_len = int(len(item_data[1])*0.8)
    train_data = item_data[:, :train_len].tolist()
    test_data = item_data[:, train_len:].tolist()
    test_time = time[train_len:]
    pred_data = np.zeros(((81, 81, 0))).tolist()
    all_zeros = []
    for i in range(81):
        all_zeros.append(True)
    for i in range(81):
        for j in range(81):
            if i == j:
                continue
            if i > 0 and (all_zeros[i] or all_zeros[j]):
                cf[i][j] = it[i][j] = sc[i][j] = sc[j][i] = 0
                continue
            # 如果i或者j有一个全为0，则不做处理，只需要在i == 0时，将所有的i,j判断一次即可
            if i == 0:
                flag1 = True
                flag2 = True
                for m in range(len(train_data[i])):
                    if train_data[i][m][0] != 0:
                        flag1 = False
                    if train_data[j][m][0] != 0:
                        flag2 = False
                    if not flag1 and not flag2:
                        break
                all_zeros[i] = flag1
                all_zeros[j] = flag2
                if flag1 or flag2:
                    cf[i][j] = it[i][j] = sc[i][j] = sc[j][i] = 0
                    continue
            lr = LinearRegression()
            lr.fit(train_data[i], train_data[j])
            #记录得分高于MIN_SCORE的模型
            sc[i][j] = lr.score(train_data[i], train_data[j])
            cf[i][j] = lr.coef_
            it[i][j] = lr.intercept_
            # if(sc[i][j] > MIN_SCORE and lr.coef_ > 0 and lr.intercept_ > 0):
            if sc[i][j] > MIN_SCORE and sc[j][i] > MIN_SCORE if i > j else True:
                #预测
                td = lr.predict(test_data[i])
                for d in td:
                    pred_data[i][j].append(d.tolist())

    # 处理时间
    tt = test_time[0]
    for i in range(len(test_time)):
        test_time[i] -= tt

    # 记录每两个度量之间的残差
    dists = np.zeros((81, 81)).tolist()
    # 获得cookDistance
    for i in range(81):
        for j in range(81):
            if sc[i][j] > MIN_SCORE and sc[j][i] > MIN_SCORE:
                y = pred_data[i][j]
                z = []
                for k in range(len(y)):
                    z.append(abs(y[k][0]-test_data[j][k][0]))
                dist = sts.getDist(z, 2, test_data[i])
                if dist == 0:
                    continue
                dists[i][j] = dist
    return test_time, sc, dists, test_data, pred_data

'''
功能：画出每两个度量的predict残差图
'''
def drawResidual(test_time, sc, item_name, dists, func_index, func_name, to_file):
    for i in range(81):
        for j in range(81):
            if sc[i][j] > MIN_SCORE and sc[j][i] > MIN_SCORE:
                x = test_time
                if dists[i][j] == 0:
                    continue
                y = dists[i][j]
                up = [np.percentile(y, 75)]*len(y)
                plt.xlabel("time(ms)")
                plt.ylabel(str(item_name[j]))
                plt.title(str(item_name[i])+'---'+str(item_name[j])+'('+func_name[int(func_index[i][j])]+')')
                plt.plot(x, y, c = 'b')
                plt.plot(x, up, c = 'r', ls = ':')
                plt.savefig(to_file+'//metrics//'+str(item_name[i])+'-'+str(item_name[j])+'.png')
                plt.close()
    return

'''
功能：画出度量之间的关联图
'''
def drawNormal(sc, to_file):
    edgelist = []
    for i in range(81):
        for j in range(81):
            if sc[i][j] > MIN_SCORE and sc[j][i] > MIN_SCORE:
                edgelist.append((i, j))
    graph = nx.empty_graph(81)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist=[i for i in range(81)],
                           node_color='b', node_size=20, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=1, alpha=0.5, edge_color='black')
    plt.savefig(to_file+'//graph//random_layout.png')
    plt.close()
    return

'''
功能：判断某个时间是否出现异常，先假设是test_time[i]
如何判定是否有异常:遍历与某个度量i有关联的所有度量j，如果dists[i][j]过大，将异常边标红
'''
def drawAnomaly(sc, dists, test_time, to_file):
    # 记录与每个度量关联的所有度量
    models = [[] for i in range(81)]
    edgelist = []
    for i in range(81):
        for j in range(81):
            if sc[i][j] > MIN_SCORE and sc[j][i] > MIN_SCORE:
                models[i].append(j)
                models[j].append(i)
                edgelist.append((i, j))
    anomaly_edge = []
    normal_edge = []
    anomaly_score = []
    cnt = 0
    for i in range(81):
        for j in range(len(models[i])):
            if i != j and dists[i][j] != 0 and dists[i][j][detect_time] > np.percentile(dists[i][j], 75):
                cnt += 1
                anomaly_edge.append((i, j))
        if (len(models[i]) > 0):
            anomaly_score.append(1.0 * cnt / len(models[i]))
        else:
            anomaly_score.append(-1)
    normal_edge = list(set(edgelist).difference(set(anomaly_edge)))
    anomaly_graph = nx.empty_graph(81)
    a_pos = nx.random_layout(anomaly_graph)
    nx.draw_networkx_nodes(anomaly_graph, a_pos, nodelist=[i for i in range(81)], node_color='b', node_size=20,
                           alpha=0.8)
    nx.draw_networkx_edges(anomaly_graph, a_pos, edgelist=anomaly_edge, width=1, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(anomaly_graph, a_pos, edgelist=normal_edge, width=1, alpha=0.5, edge_color='black')
    plt.savefig(to_file+'//graph//anomaly-'+str(test_time[detect_time])+'.png')

    # 将异常检测数据存入文件
    with open(to_file+'//anomaly_data-'+ str(test_time[detect_time])+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time: ' + str(test_time[detect_time])])
        writer.writerow(['x', 'y', 'anomaly_score'])
        for i in range(len(anomaly_edge)):
            writer.writerow([item_name[anomaly_edge[i][0]], item_name[anomaly_edge[i][1]], anomaly_score[anomaly_edge[i][1]]])
    return


# 从文件读到度量名称、度量数据
item_name, item_data, time = sts.loadData('data.csv')

# SLR
test_time, sc_slr, dists_slr, test_data, pred_data_slr = linearReg(item_name, item_data, time)

# SLRT
'''
    通过对数据进行三种预处理方式(log, reciprocal, sqrt)，根据预测得分找出最优的处理方式
'''
item_data_log = sts.logarithm(item_data)
test_time, sc_slrt_log, item_name, dists_slrt_log, test_data,\
pred_data_slrt_log = linearReg(item_name, item_data_log, time)

item_data_rec = sts.reciprocal(item_data)
test_time, sc_slrt_rec, item_name, dists_slrt_rec, test_data,\
pred_data_slrt_rec = linearReg(item_name, item_data_rec, time)

item_data_sqrt = sts.square_root(item_data)
test_time, sc_slrt_sqrt, item_name, dists_slrt_sqrt, test_data,\
pred_data_slrt_sqrt = linearReg(item_name, item_data_sqrt, time)

#SLRS
item_data_smooth = sts.smooth(item_data, K)
test_time, sc_slrs, item_name, dists_slrs, test_data,\
pred_data_slrs = linearReg(item_name, item_data_smooth, time)

#通过对比slr和slrt的三种处理方式以及slrs对应的r2，找到最适合的模型
func_index = np.zeros((82, 82)).tolist()
func_name = ['none', 'slr', 'slrt_log', 'slrt_rec', 'slrt_sqrt', 'slrs']
sc = np.zeros((82, 82)).tolist()
dists = np.zeros((82, 82)).tolist()
for i in range(81):
    for j in range(81):
        r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
        if sc_slr[i][j] > MIN_SCORE and sc_slr[i][j] > MIN_SCORE \
                and len(test_data[j]) == len(pred_data_slr[i][j]):
            r1 = sts.r2(test_data[j], pred_data_slr[i][j])
        if sc_slrt_log[i][j] > MIN_SCORE and sc_slrt_log[i][j] > MIN_SCORE \
                and len(test_data[j]) == len(pred_data_slrt_log[i][j]):
            r2 = sts.r2(test_data[j], pred_data_slrt_log[i][j])
        if sc_slrt_rec[i][j] > MIN_SCORE and sc_slrt_rec[i][j] > MIN_SCORE \
                and len(test_data[j]) == len(pred_data_slrt_rec[i][j]):
            r3 = sts.r2(test_data[j], pred_data_slrt_rec[i][j])
        if sc_slrt_sqrt[i][j] > MIN_SCORE and sc_slrt_sqrt[i][j] > MIN_SCORE\
                and len(test_data[j]) == len(pred_data_slrt_sqrt[i][j]):
            r4 = sts.r2(test_data[j], pred_data_slrt_sqrt[i][j])
        if sc_slrs[i][j] > MIN_SCORE and sc_slrs[i][j] > MIN_SCORE \
                and len(test_data[j]) == len(pred_data_slrs[i][j]):
            r5 = sts.r2(test_data[j], pred_data_slrs[i][j])
        maxr = max([r1, r2, r3, r4, r5])
        if r1 == maxr:
            if sc[i][j] < sc_slr[i][j]:
                sc[i][j] = sc_slr[i][j]
                dists[i][j] = dists_slr[i][j]
                func_index[i][j] = 1
        if r2 == maxr:
            if sc[i][j] < sc_slrt_log[i][j]:
                sc[i][j] = sc_slrt_log[i][j]
                dists[i][j] = dists_slrt_log[i][j]
                func_index[i][j] = 2
        if r3 == maxr:
            if sc[i][j] < sc_slrt_rec[i][j]:
                sc[i][j] = sc_slrt_rec[i][j]
                dists[i][j] = dists_slrt_rec[i][j]
                func_index[i][j] = 3
        if r4 == maxr:
            if sc[i][j] < sc_slrt_sqrt[i][j]:
                sc[i][j] = sc_slrt_sqrt[i][j]
                dists[i][j] = dists_slrt_sqrt[i][j]
                func_index[i][j] = 4
        if r5 == maxr:
            if sc[i][j] < sc_slrs[i][j]:
                sc[i][j] = sc_slrs[i][j]
                dists[i][j] = dists_slrs[i][j]
                func_index[i][j] = 5


warnings.filterwarnings('ignore')
drawResidual(test_time, sc_slr, item_name, dists_slr, func_index, func_name, 'slr')
drawNormal(sc_slr, 'slr')
drawAnomaly(sc_slr, dists_slr, test_time, 'slr')