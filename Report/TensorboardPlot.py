from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import scipy.signal as signal

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def smooth(y, weight=0.9):
    
    last = y[0]
    smoothed = []
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    y = np.array(smoothed)
    return y

if __name__ == '__main__':

    tensorboard_log_file = './events.out.tfevents.1552975068.DESKTOP-7PQVU8O'
    key_str = 'losses/mask'
    ea=event_accumulator.EventAccumulator(tensorboard_log_file) 
    ea.Reload()
    print(ea.scalars.Keys())
    val_sin=ea.scalars.Items(key_str) 
    y_sin = [i.value for i in val_sin]
    x_sin = [i.step-132000 for i in val_sin]
    # x_sin =[i for i in range(len(y_sin))]
    y_sin =smooth(y_sin,0.5)
    # fig=plt.figure()
    # # ax1=fig.plot()
    # # plt.plot(x,y,label=key_str1)

    # #开启一个窗口，num设置子图数量，figsize设置窗口大小，dpi设置分辨率
    # fig = plt.figure(num=1, figsize=(500, 500),dpi=80)
    # #直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列   
    # # plt.plot(x,y1)

    # # plt.title('Generator Convergence Speed Analysis')
    # plt.plot(x_sin,y_sin,label='spatial semantic loss')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('iteration times')
    # plt.ylabel('loss')
    # #显示绘图结果
    # plt.show()
    # fig=plt.figure()
    # ax1=fig.plot()
    # plt.plot(x,y,label=key_str1)

    #开启一个窗口，num设置子图数量，figsize设置窗口大小，dpi设置分辨率
    figsize = 11,9
    figure, ax = plt.subplots(figsize=figsize)
    # figure, ax = plt.figure(num=1, figsize=(500, 500),dpi=80)
    #直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列   
    # plt.plot(x,y1)
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 23,
    }
    # plt.title('Generator Convergence Speed Analysis')
    A,=plt.plot(x_sin,y_sin,label='spatial semantic loss')
    plt.legend()
    plt.grid()
    legend = plt.legend(handles=[A,],prop=font1)
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel('Iteration times',font1)
    plt.ylabel('Loss',font1)
    #显示绘图结果
    plt.show()


    #加载日志数据
    tensorboard_log_file = './events.out.tfevents.1553090854.cxh-Z270X-Gaming-K5'
    tensorboard_log_file1 = './events.out.tfevents.1553086045.ubuntu-Z270X-Gaming-K5'
    key_str1 = 'data/g_total_loss'
    key_str2 = 'data/mask_loss'
    key_str3 = 'data/mask_sparse'
    key_str = 'data/g_loss_fake'
    ea1=event_accumulator.EventAccumulator(tensorboard_log_file) 
    ea1.Reload()
    ea2=event_accumulator.EventAccumulator(tensorboard_log_file1) 
    ea2.Reload()
    # print(ea.scalars.Keys())

    val_1=ea1.scalars.Items(key_str1)   
    val_2=ea1.scalars.Items(key_str2)
    val_3=ea1.scalars.Items(key_str3)
    y1 = [i.value for i in val_1]
    y2 = [i.value for i in val_2]
    y3 = [i.value for i in val_3]
    x = [i.step for i in val_1]


    val_4=ea2.scalars.Items(key_str)
    x2 = [i.step for i in val_4]
    x2 = [200,]+x2
    x2 += [42000,43000]
    y4 = [i.value for i in val_4]
    y4 = [41.7,] + y4
    y4 = y4 + [9.7,10.2]
    y4 =smooth(y4,0.95)
    # print(y4)
    # print(x2)
    # f1=interp1d(x2,y4,kind='linear')#线性插值
    # x_tmp = x[2:]
    # # x_tmp[0] = 600
    # print(x_tmp)
    # y4 = f1(x)
    # print(y4)
    # f2=interp1d(x,y,kind='cubic')#三次样条插值

    y5 = []
    for i in range(len(x)):
        y5.append(y1[i] - y2[i] * 100 - y3[i])
        

    # print(x)
    # y = np.array(y)
    # x = signal.medfilt(x,3)
    # y1 = smooth(y1)
    y5 = smooth(y5,0.95)
    # weight=0.9
    # last = y[0]
    # smoothed = []
    # for point in y:
    #     smoothed_val = last * weight + (1 - weight) * point
    #     smoothed.append(smoothed_val)
    #     last = smoothed_val
    # y = np.array(smoothed)
    # f1=interp1d(x,y,kind='linear')#线性插值
    # f2=interp1d(x,y,kind='cubic')#三次样条插值
    
    fig=plt.figure()
    # ax1=fig.plot()
    # plt.plot(x,y,label=key_str1)

    #开启一个窗口，num设置子图数量，figsize设置窗口大小，dpi设置分辨率
    figsize = 11,9
    figure, ax = plt.subplots(figsize=figsize)
    # figure, ax = plt.figure(num=1, figsize=(500, 500),dpi=80)
    #直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列   
    # plt.plot(x,y1)
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 30,
    }
    # plt.title('Generator Convergence Speed Analysis')
    A,=plt.plot(x,y5,label='G loss with the ASP Module', linewidth=2)
    B,=plt.plot(x2,y4,label='G loss without the ASP Module', linewidth=2)
    plt.legend()
    plt.grid()
    legend = plt.legend(handles=[A,B],prop=font1)
    plt.tick_params(labelsize=28)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel('Iteration times',font1)
    plt.ylabel('Loss',font1)
    #显示绘图结果
    plt.show()