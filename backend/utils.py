import numpy as np
import os
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path , delimiter=',') 
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:] 
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY) 
        path_XYs.append(XYs)
    return path_XYs

def plot(path_XYs, points=[]):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XY in enumerate(path_XYs):
        ax.plot(XY[:, 0], XY[:, 1], linewidth=2, label=f"Curve {i}")
        first_point = XY[0]
        # ax.plot(first_point[0], first_point[1], 'mo')
        if points != []:
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            ax.scatter(x, y)
    ax.set_aspect('equal')
    ax.legend()
    # Save combined image
    plt.savefig("images/final.png")
    plt.show()
