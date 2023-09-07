import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import get_cmap
import matplotlib.cm
import matplotlib.pyplot as plt

cmap = get_cmap('binary_r')
norm = matplotlib.colors.Normalize(vmin=0, vmax=1) 
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)    

def plot_adjacency(C,h,ax):

    # Plot

    hcum = np.cumsum(h)
    N = len(h)

    for i in range(N):
        for j in range(N):
            rect = plt.Rectangle((hcum[i]-h[i], hcum[j]-h[j]), h[i], h[j], fill=True, facecolor=cmap(norm(C[i,j])), edgecolor='r', linewidth=1)
            ax.add_patch(rect)

    ax.invert_yaxis()
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    
    return sm


