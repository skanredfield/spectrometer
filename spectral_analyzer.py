import cv2
from cv2 import Mat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class SpectralAnalyzer:

    def analyze(self, img: Mat):
        # convert from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        shape = img.shape
        r_dist = []
        b_dist = []
        g_dist = []
        i_dist = []
        print(shape)
        for i in range(shape[1]):
            r_val = np.mean(img[:, i][:, 0])
            g_val = np.mean(img[:, i][:, 1])
            b_val = np.mean(img[:, i][:, 2])
            i_val = (r_val + g_val + b_val) / 3

            r_dist.append(r_val)
            g_dist.append(g_val)
            b_dist.append(b_val)
            i_dist.append(i_val)
        
        plt.subplot(2, 1, 1)
        plt.imshow(img, interpolation='nearest', aspect='auto')

        r_peaks, _ = find_peaks(r_dist, height=0)
        g_peaks, _ = find_peaks(g_dist, height=0)
        b_peaks, _ = find_peaks(b_dist, height=0)

        plt.subplot(2, 1, 2)

        plt.plot(r_dist, color='r', label='red')
        # plt.plot(r_peaks, [r_dist[i] for i in r_peaks], '*')
        plt.vlines(r_peaks, ymin=min(r_dist), ymax=max(r_dist))
        plt.plot(g_dist, color='g', label='green')
        plt.vlines(g_peaks, ymin=min(g_dist), ymax=max(g_dist))
        plt.plot(b_dist, color='b', label='blue')
        plt.vlines(b_peaks, ymin=min(b_dist), ymax=max(b_dist))
        plt.plot(i_dist, color='k', label='mean')
        plt.margins(x=0)

        plt.legend(loc="upper left")
        plt.savefig("output/result.png")
        plt.show()
