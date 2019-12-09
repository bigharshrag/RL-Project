import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import legend

def lamdba_vary():
    lams = {}
    with open('sarsa_lamdbavary2') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in lams:
                lams[float(x[0])] = []
            lams[float(x[0])].append(float(x[1]))
        
        x_ax = []
        y_ax = []
        for lam in lams.keys():
            x_ax.append(lam)
            y_ax.append(np.mean(lams[lam]))
        plt.plot(x_ax, y_ax)
        plt.show()

# lamdba_vary()

def speed_analysis():
    slam_o = {}
    slam_p = {}
    tamerrl_p = {}
    demorl = {}
    detarl = {}
    with open('exp_sarsalam') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in slam_o:
                slam_o[float(x[0])] = []
            slam_o[float(x[0])].append(float(x[1]))
    
    with open('exp_sarsalam_pess') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in slam_p:
                slam_p[float(x[0])] = []
            slam_p[float(x[0])].append(float(x[1]))

    with open('exp_tamerrl_pess') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in tamerrl_p:
                tamerrl_p[float(x[0])] = []
            tamerrl_p[float(x[0])].append(float(x[1]))
    
    with open('exp_demorl_pess') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in demorl:
                demorl[float(x[0])] = []
            demorl[float(x[0])].append(float(x[1]))

    with open('exp_demotamerrl4_pess') as f:
        for l in f:
            x = l.split()
            if float(x[0]) not in detarl:
                detarl[float(x[0])] = []
            detarl[float(x[0])].append(float(x[1]))
    
    x_ax = []
    y_slam_o = [[], []]
    y_slam_p = [[], []]
    y_trl = [[], []]
    y_drl = [[], []]
    y_dtr = [[], []]
    for k in slam_o:
        x_ax.append(k)
        y_slam_o[0].append(np.mean(slam_o[k]))
        y_slam_p[0].append(np.mean(slam_p[k]))
        y_trl[0].append(np.mean(tamerrl_p[k]))
        y_drl[0].append(np.mean(demorl[k]))
        y_dtr[0].append(np.mean(detarl[k]))

        y_slam_o[1].append(np.std(slam_o[k]))
        y_slam_p[1].append(np.std(slam_p[k]))
        y_trl[1].append(np.std(tamerrl_p[k]))
        y_drl[1].append(np.std(demorl[k]))
        y_dtr[1].append(np.std(detarl[k]))

    # print(y_slam_o[0])
    # print(y_slam_p[0])
    # print(y_trl[0])
    # print(y_drl[0])
    # print(y_dtr[0])

    bar_x = ['SARSA($\lambda$) (opt.)', 'SARSA($\lambda$) (pess.)', 'TAMER+RL', 'Demos+RL', 'Proposed']
    bar_y = [y_slam_o[0][-1], y_slam_p[0][-1], y_trl[0][-1], y_drl[0][-1], y_dtr[0][-1]]
    bar_yerr = [y_slam_o[1][-1], y_slam_p[1][-1], y_trl[1][-1], y_drl[1][-1], y_dtr[1][-1]]

    plt.bar(np.arange(len(bar_x)), bar_y, yerr=bar_yerr, ecolor='black', capsize=2)
    plt.xticks(np.arange(len(bar_x)), ('SARSA($\lambda$)(o.)', 'SARSA($\lambda$)(p.)', 'TAMER+RL', 'Demos+RL', 'Proposed'))
    axes = plt.gca()
    axes.set_ylim([-170, -90])
    plt.show()

    plt.errorbar(x_ax, y_slam_o[0], yerr=y_slam_o[1], label='Sarsa($\lambda$) (opt.)', fmt='.-g', elinewidth=0.5, ecolor='grey', capsize=2, linewidth=1.5)
    plt.errorbar(x_ax, y_slam_p[0], yerr=y_slam_p[1], label='Sarsa($\lambda$) (pess.)', fmt='.-m', elinewidth=0.5, ecolor='grey', capsize=2, linewidth=1.5)
    plt.errorbar(x_ax, y_trl[0], yerr=y_trl[1], label='TAMER+RL', fmt='.-y', elinewidth=0.5, ecolor='grey', capsize=2, linewidth=1.5)
    plt.errorbar(x_ax, y_drl[0], yerr=y_drl[1], label='Demos+RL', fmt='.-r', elinewidth=0.5, ecolor='grey', capsize=2, linewidth=1.5)
    plt.errorbar(x_ax, y_dtr[0], yerr=y_dtr[1], label='Proposed', fmt='.-c', elinewidth=0.5, ecolor='grey', capsize=2, linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Average reward in 100 runs')
    legend()
    plt.show()

speed_analysis()