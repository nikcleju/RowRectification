import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import skimage as ski

from itertools import pairwise

from .ridge import Ridge


class RidgeExtractor:
    def __init__(self) -> None:
        self.gaussflt_sigma = [5,50]

    def get_image_kernel(self, shape):
        Idirac = np.zeros(shape)
        Idirac[shape[0]//2,shape[1]//2] = 1
        Ikernel = ski.filters.gaussian(Idirac, sigma=self.gaussflt_sigma)
        #plt.imshow(Ikernel, cmap='gray')
        return Ikernel

    def preprocess_image(self, I, return_intermediates=False):

        Igray = ski.color.rgb2gray(I)
        #plt.imshow(I, cmap='gray')

        # Invert image
        maxval = np.max(Igray)
        Iinv = maxval - Igray

        # Gaussian filtering
        Ifilt = ski.filters.gaussian(Iinv, sigma=self.gaussflt_sigma)

        if return_intermediates:
            return Ifilt, Iinv, Igray
        else:
            return Ifilt

    def _find_starting_points(self, I):
        # Find starting points from a middle band
        band_mid = int(I.shape[1] / 2)
        band_halfwidth = 50
        hist = np.sum(I[:, band_mid-band_halfwidth:band_mid+band_halfwidth], axis=1)
        peaks, _ = scipy.signal.find_peaks(hist)

        return peaks, hist

    def get_image_middleband_peaks(self, I):
        # Find starting points from a middle band
        peaks, hist = self._find_starting_points(I)

        fig, ax = plt.subplots()
        ax.plot(hist)
        ax.plot(peaks, hist[peaks], "o")

        return fig

    def extract_ridges(self, I, return_plot=False):

        # Find starting points from a middle band
        peaks, _ = self._find_starting_points(I)

        # Follow ridges and plot
        band_mid = int(I.shape[1] / 2)  # TODO: make in class
        ridges = [self._ridge_following(I, [peak, band_mid], stop_thresh=0.5) for peak in peaks]

        return ridges

    @staticmethod
    def get_image_with_ridges(ridges, I=None, shape=None, fg_color=0, bg_color=1, line=False):

        # Prepare background image
        if I is not None:
            Iout = np.copy(I)
        if I is None and shape is not None:
            Iout = np.zeros(shape) + bg_color
        elif I is None and shape is None:
            max_x = max([max([x for x in ridge.x]) for ridge in ridges])
            max_y = max([max([y for y in ridge.y]) for ridge in ridges])
            Iout = np.zeros((max_y, max_x))  + bg_color

        # Plot ridges on image
        for ridge in ridges:
            if line:
                for p1, p2 in pairwise(ridge):
                    Iout = cv2.line(Iout, list(map(int,p1)), list(map(int,p2)), color=fg_color)
            else:
                for x,y in ridge:
                    Iout[int(y), int(x)] = fg_color

        return Iout

    @staticmethod
    def _ridge_following( I, start_point, max_stride=1, stop_thresh = 0.5):
        """Follow a ridge in the image, starting from a given point, going both left and right

        Args:
            I: the image
            start_point: one point on the ridge
            max_stride (int, optional): local window where to search for the next point on the ridge. Defaults to 1.
            stop_thresh (float, optional): stop ridge when next point is smaller than (stop_thresh * max value) on the ridge

        Returns:
            a vector with the row coordinates of points on the ridge
        """

        assert len(I.shape)==2
        cols = I.shape[1]

        start_row, start_col = start_point

        # Prepare output vector, start with -1
        ridge = -1 * np.ones(cols, dtype=int)
        ridge[start_col] = start_row

        # Follow ridge to the right of start_point
        max_value = I[start_row, start_col]
        for i in range(start_col+1, cols):
            from_row = max(ridge[i-1] - max_stride, 0)
            to_row = min(ridge[i-1] + max_stride + 1, I.shape[0])
            ridge[i] = from_row + np.argmax(I[from_row : to_row, i])

            # Update max value
            max_value = max(max_value, I[ridge[i],i])

            # Check termination
            if I[ridge[i],i] < stop_thresh * max_value:
                ridge[i] = -1
                break

        # Follow ridge to the left of start_point
        max_value = I[start_row, start_col]
        for i in range(start_col-1, -1, -1):
            from_row = max(ridge[i+1] - max_stride, 0)
            to_row = min(ridge[i+1] + max_stride + 1, I.shape[0])
            ridge[i] = from_row + np.argmax(I[from_row : to_row, i])

            # Update max value
            max_value = max(max_value, I[ridge[i],i])

            # Check termination
            if I[ridge[i],i] < stop_thresh * max_value:
                ridge[i] = -1
                break

        # Right now, ridge is as wide as the image, with -1 where it is missing
        #Imge Keep only the actual part (!= -1)
        #return ridge
        x = [i        for i in range(len(ridge)) if ridge[i] != -1]
        y = [ridge[i] for i in range(len(ridge)) if ridge[i] != -1]
        return Ridge(x,y)

