import matplotlib.pyplot as plt
import numpy as np

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

lamdba_vary()