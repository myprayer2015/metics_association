import numpy as np
import csv
# 存放一些数据处理函数，以及一些功能函数

'''
loadData返回值:
    item_data是(81*1440)的二维数组
    time是从0开始刻度为1的时间值
'''
def loadData(dataFile):
    item_name = []
    item_data = np.zeros((81, 0)).tolist()
    time = []

    reader = csv.reader(open(dataFile, 'r'))
    for item in reader:
        if reader.line_num == 1:
            item_name = item[1:]
            # print(item_name)
        elif reader.line_num > 1441:
            break
        else:
            for i in range(len(item)):
                t = []
                if i == 0:
                    time.append(int(item[i]))
                    continue
                if item[i] == '':
                    t.append(0)
                else:
                    t.append(float(item[i]))
                item_data[i-1].append(t)
    for i in range(len(item_name)):
        if item_name.count(item_name[i]) > 1:
            item_name[i] += str(item_name.count(item_name[i]))
    ft = time[0]
    for i in range(1, len(time)):
        time[i] = int((time[i] - ft) / 60)
    time[0] = 0
    for i in range(0, len(time)):
        time[i] *= 3.5
    return item_name, item_data, time

#每一行是一个时刻的所有metrics值
def loadLassoData(dataFile):
    item_name = []
    item_data = np.zeros((1500, 81)).tolist()
    time = []

    reader = csv.reader(open(dataFile, 'r'))
    for item in reader:
        if reader.line_num == 1:
            item_name = item
        elif reader.line_num > 1441:
            break
        else:
            t = []
            for i in range(len(item)):
                if i == 0:
                    time.append(int(item[i]))
                    continue
                if item[i] == '':
                    t.append(0)
                else:
                    t.append(float(item[i]))
            item_data[reader.line_num-2] = t
    for i in range(len(item_name)):
        if item_name.count(item_name[i]) > 1:
            item_name[i] += str(item_name.count(item_name[i]))
    ft = time[0]
    for i in range(1, len(time)):
        time[i] = int((time[i] - ft) / 60)
    time[0] = 0
    for i in range(0, len(time)):
        time[i] *= 3.5
    return item_name, item_data, time

def getDist(residual, p, x_value):
    d = cookDist(residual, p, x_value)
    return d

def cookDist(residual, p, x_value):
    n = len(x_value)
    #残差的方差
    rv = np.var(residual)
    #x的均值
    xm = np.mean(x_value)
    #x的方差
    xv = np.var(x_value)
    d = []
    for i in range(len(residual)):
        dl = 1.0*residual[i]**2/(p*rv)
        h = 1.0/n + (x_value[i]-xm)**2/((n-1)*xv)
        dr = 1.0*h/(1-h)**2
        d.append(float(dl*dr))
    return d

def r2(test_data, pred_data):
    sum = 0
    for i in range(len(test_data)):
        sum += test_data[i][0]
    mean = 1.0*sum/len(test_data)
    up, down = 0, 0
    # print(len(test_data), len(pred_data))
    for i in range(len(test_data)):
        up += (test_data[i][0]-pred_data[i][0])**2
        down += (test_data[i][0]-mean)**2
    return 1.0-1.0*up/down if down != 0 else 0

def logarithm(x):
    ret = []
    for i in range(len(x)):
        t = []
        for j in range(len(x[i])):
            if x[i][j][0] != 0:
                t.append([np.log(x[i][j][0])])
            else:
                t.append([0])
        ret.append(t)
    return ret

def reciprocal(x):
    ret = []
    for i in range(len(x)):
        t = []
        for j in range(len(x[i])):
            t.append([1.0/(1+x[i][j][0])])
        ret.append(t)
    return ret

def square_root(x):
    ret = []
    for i in range(len(x)):
        t = []
        for j in range(len(x[i])):
            t.append([np.sqrt(x[i][j][0])])
        ret.append(t)
    return ret

def smooth(x, k):
    ret = []
    for i in range(len(x)):
        t = 0
        r = []
        for j in range(len(x[i])):
            if j >= k-1:
                t += x[i][j][0]
                if j > k-1:
                    t -= x[i][j-k][0]
                if k != 0:
                    r.append([1.0*t/k])
                else:
                    r.append([x[i][j][0]])
            else:
                t += x[i][j][0]
                r.append([x[i][j][0]])
        ret.append(r)
    return ret

# about arx_lre
'''
给模型打分
参数y,y1分别的是y的真实值和预测值
'''
def fitness_score(y, y1):
    # print(len(y), len(y1))
    # print(y)
    # print(y1)
    ym = np.mean(y)
    up, down = 0.0, 0.0
    for i in range(len(y)):
        up += 1.0*(y[i]-y1[i])**2
        down += 1.0*(y[i]-ym)**2
    ret = (1-np.sqrt(up/down))*100
    return ret

# 模型的置信度
def confidence_score(pre_cs, k, fc):
    return 1.0*(pre_cs*(k-1)+fc)/k

