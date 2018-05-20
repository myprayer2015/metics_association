import statistics_func as sts
import numpy as np
import warnings
from sklearn.linear_model import Lasso, LassoCV

'''
功能：构建lassoCV模型
'''
def lassoReg(tmp_data, time):
    item_data = np.array(tmp_data)
    train_len = int(len(item_data) * 0.8)
    train_data = item_data[:train_len, :]
    test_data = item_data[train_len:, :]
    test_time = time[train_len:]
    all_zeros = []
    lc_models = []

    tt = test_time[0]
    for i in range(len(test_time)):
        test_time[i] -= tt

    # lassoCV_score 最高为1，可能为负，如果某个数据全为0，则将score置为100，表示不做预测
    lc_scores = np.zeros(81).tolist()
    for i in range(81):
        y1 = train_data[:, i:i+1]
        flag = True
        for k in range(len(y1)):
            if y1[k][0] != 0:
                flag = False
                break
        y = y1.ravel().tolist()
        all_zeros.append(flag)
        if flag == True:
            lc_models.append(0)
            lc_scores[i] = 100
            continue
        x = train_data[:, :i].tolist()
        r = train_data[:, i+1:].tolist()
        for j in range(len(x)):
            x[j].extend(r[j])
        lc = LassoCV().fit(x, y)
        test_x = test_data[:, :i].tolist()
        test_y = test_data[:, i:i+1].tolist()
        tr = test_data[:, i+1:]
        for j in range(len(test_x)):
            test_x[j].extend(tr[j])
        lc_scores[i] = lc.score(test_x, test_y)
    return lc_models, lc_scores, test_time, test_data

warnings.filterwarnings('ignore')
item_name, item_data, time = sts.loadLassoData('data.csv')
lc_models, lc_scores, test_time, test_data = lassoReg(item_data, time)