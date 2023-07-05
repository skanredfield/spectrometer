import cv2
from cv2 import Mat
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from spectral_utils import wavelength_to_rgb, np_wavelength_to_rgb
from dft import *


class SpectralAnalyzer:

    def analyze(self, img: Mat):
        # convert from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)

        shape = img.shape
        r_dist = []
        b_dist = []
        g_dist = []
        i_dist = []

        for i in range(shape[1]):
            r_val = np.mean(img[:, i][:, 0])
            g_val = np.mean(img[:, i][:, 1])
            b_val = np.mean(img[:, i][:, 2])
            i_val = (r_val + g_val + b_val) / 3

            r_dist.append(r_val)
            g_dist.append(g_val)
            b_dist.append(b_val)
            i_dist.append(i_val)
        
        # fix the right side of the image, as the camera brightens the right side
        r_scale_fix = np.linspace(1.0, 0.5, len(r_dist))
        g_scale_fix = np.linspace(1.0, 0.5, len(g_dist))
        b_scale_fix = np.linspace(1.0, 0.5, len(b_dist))
        r_dist = r_dist * r_scale_fix
        g_dist = g_dist * g_scale_fix
        b_dist = b_dist * b_scale_fix
        
        min_line_height = min(min(r_dist), min(g_dist), min(b_dist))
        max_line_height = max(max(r_dist), max(g_dist), max(b_dist))
        r_dist_normalized = r_dist / max_line_height
        g_dist_normalized = g_dist / max_line_height
        b_dist_normalized = b_dist / max_line_height

        # height value serves as the threshold on the amplitude of the peak and is picked by trial and error
        r_peaks, _ = find_peaks(r_dist_normalized, height=0.5)
        g_peaks, _ = find_peaks(g_dist_normalized, height=0.5)
        b_peaks, _ = find_peaks(b_dist_normalized, height=0.5)

        # Important! r_peaks, g_peaks, b_peaks are the indices of peaks!

        r_peaks = np.array(self._remap_range(r_peaks, 589, 750))
        g_peaks = np.array(self._remap_range(g_peaks, 490, 570))
        b_peaks = np.array(self._remap_range(b_peaks, 430, 490))

        self._plot_interference_pattern(img)
        self._plot_rgb_curves(r_dist_normalized, g_dist_normalized, b_dist_normalized)
        self._plot_spectrum_black_lines(min_line_height, max_line_height, r_peaks, g_peaks, b_peaks)
        ylim_bottom, ylim_top = plt.ylim()
        self._plot_emission_lines(min_line_height, max_line_height, r_peaks, g_peaks, b_peaks, ylim_bottom, ylim_top)


        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
        
        plt.savefig("output/result.png")
        plt.show()
        

    def _plot_interference_pattern(self, img):
        ax1 = plt.subplot(4, 1, 1)
        plt.imshow(img, interpolation='nearest', aspect='auto')

        # Warning! The angles are picked for the prototype spectrometer and to match the sodium line based
        # on the standard DVD grating spacing of 0.74 micron
        start_angle_rad = 0
        end_angle_rad = 1.27

        ax2 = ax1.twiny()
        ax2.plot(np.linspace(start_angle_rad, end_angle_rad, 20), np.ones(20))
        ax2.set_xlim([start_angle_rad, end_angle_rad])
        ax2.set_xlabel(r"Angle, \phi (rad)")

    def _plot_rgb_curves(self, r_dist_normalized, g_dist_normalized, b_dist_normalized):
        ax = plt.subplot(4, 1, 2)

        plt.plot(r_dist_normalized, color='r')
        # plt.plot(r_peaks, [r_dist[i] for i in r_peaks], '*')
        plt.plot(g_dist_normalized, color='g')
        plt.plot(b_dist_normalized, color='b')
        # plt.plot(i_dist, color='k', label='mean')
        plt.margins(x=0)

        # show peaks
        self._annotate_max(ax, ax.lines[0].get_xdata(), r_dist_normalized, g_dist_normalized, b_dist_normalized)

    def _plot_spectrum_black_lines(self, min_line_height, max_line_height, r_peaks, g_peaks, b_peaks):
        plt.subplot(4, 1, 3)
        self._draw_spectrum((min_line_height, max_line_height))

        plt.vlines(r_peaks, ymin=min_line_height, ymax=max_line_height, color='black')
        plt.vlines(g_peaks, ymin=min_line_height, ymax=max_line_height, color='black')
        plt.vlines(b_peaks, ymin=min_line_height, ymax=max_line_height, color='black')

    def _plot_emission_lines(self, min_line_height, max_line_height, r_peaks, g_peaks, b_peaks, ylim_bottom, ylim_top):
        subplot4 = plt.subplot(4, 1, 4)
        # invert color to show emission lines
        plt.vlines(r_peaks, ymin=min_line_height, ymax=max_line_height, color=np_wavelength_to_rgb(r_peaks))
        plt.vlines(g_peaks, ymin=min_line_height, ymax=max_line_height, color=np_wavelength_to_rgb(g_peaks))
        plt.vlines(b_peaks, ymin=min_line_height, ymax=max_line_height, color=np_wavelength_to_rgb(b_peaks))
        plt.xlim([380, 750])
        plt.ylim([ylim_bottom, ylim_top])
        subplot4.set_facecolor("black")




    def _draw_spectrum(self, y_range=(-1,-1)):
        min_wavelength = 380
        max_wavelength = 750
        clim = (min_wavelength, max_wavelength)
        norm = plt.Normalize(*clim)
        wl = np.arange(clim[0],clim[1]+1,2)
        colorlist = list(zip(norm(wl),[wavelength_to_rgb(w) for w in wl]))
        spectralmap = colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

        wavelengths = np.linspace(min_wavelength, max_wavelength, 1000)

        y = np.linspace(0, 6, 100)
        X,Y = np.meshgrid(wavelengths, y)

        y_min = np.min(y) if y_range[0] == -1 else y_range[0]
        y_max = np.max(y) if y_range[1] == -1 else y_range[1]
        extent = (np.min(wavelengths), np.max(wavelengths), y_min, y_max)
        plt.imshow(X, clim=clim,  extent=extent, cmap=spectralmap, aspect='auto')


    def _annotate_max(self, ax, x, y_r, y_g, y_b):
        xmax_r = x[np.argmax(y_r)]
        ymax_r = max(y_r)
        xmax_g = x[np.argmax(y_g)]
        ymax_g = max(y_g)
        xmax_b = x[np.argmax(y_b)]
        ymax_b = max(y_b)
        text = '\n'.join((
            r'Maxima',
            r'R: (x=%.3f, y=%.3f)' % (xmax_r, ymax_r),
            r'G: (x=%.3f, y=%.3f)' % (xmax_g, ymax_g),
            r'B: (x=%.3f, y=%.3f)' % (xmax_b, ymax_b)
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.01, 0.01, text, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', bbox=props)

    def _remap_range(self, values, new_min, new_max):
        try:
            old_min = min(values)
            old_max = max(values)
            if old_min == old_max: # this is the case when we get only one peak
                old_range = 1
            else:
                old_range = (old_max - old_min)  
            new_range = (new_max - new_min)  

            return [(((x - old_min) * new_range) / old_range) + new_min for x in values]
        except ValueError:
            return []
