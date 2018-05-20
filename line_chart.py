import matplotlib.pyplot as plt
def drawChart(x, y, mk, lb):
    plt.xlabel('time(ms)')
    plt.ylabel('delay(m)')
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker=mk[i], label=lb[i])
    plt.legend()
    plt.show()
    plt.close()
    return

def drawSubChart(x, y, mk, lb, tl):
    index = 221
    for i in range(4):
        plt.subplot(index+i)
        for j in range(len(x[i])):
            plt.plot(x[i][j], y[i][j], marker=mk[i][j], label=lb[i][j])
            plt.title(tl[i][j])
            plt.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return

x = [[1,2,3,4,5], [1,2,3,4,5]]
y = [[2,1,5,3,4], [5,2,1,4,3]]
marker = ['o', 'x']
lb = ['1', '2']

x1 = [[[1,2,3,4,5]], [[1,2,3,4,5]], [[1,2,3,4,5]], [[1,2,3,4,5]]]
y1 = [[[2,1,5,3,4]], [[5,2,1,4,3]], [[5,2,1,4,3]], [[5,2,1,4,3]]]
marker1 = [['o'], ['x'], ['*'], ['+']]
lb1 = [['1'], ['2'], ['3'], ['4']]
title1 = [['1'], ['2'], ['3'], ['4']]

drawChart(x, y, marker, lb)
drawSubChart(x1, y1, marker1, lb1, title1)