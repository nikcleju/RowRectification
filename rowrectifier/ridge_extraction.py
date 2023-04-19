# Copyright (c) 2023 Nicolae Cleju <ncleju@etti.tuiasi.ro>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import skimage as ski

from itertools import pairwise

from typing import Tuple, List, Sequence, Iterable, \
    Iterator, Generator, Optional, Union, Any
from numpy.typing import ArrayLike, NDArray


from .ridge import Ridge


class RidgeExtractor:
    """Extract ridges from images.

    Method:

    1. Convert image to grayscale
    2. Invert the image
    3. Gaussian filtering
    4. Identify peaks inside a band in the middle of the image
    5. Follow ridges to the right and left
    """
    def __init__(self, gaussflt_sigma: Sequence = [5,50]) -> None:
        """Constructor

        Parameters
        ----------
        gaussflt_sigma : Sequence, optional
            x and y sigma values for Gaussian filtering, by default [5,50]
        """
        self.gaussflt_sigma = gaussflt_sigma

    def get_image_kernel(self, shape: Sequence) -> NDArray:
        """Return an image displaying the kernel (i.e. impulse response)
        used in gaussian filtering.

        Parameters
        ----------
        shape : Sequence
            The desired image size (height, width)

        Returns
        -------
        NDArray
            Image with the kernel of Gaussian filtering.
        """
        Idirac = np.zeros(shape)
        Idirac[shape[0]//2,shape[1]//2] = 1
        Ikernel = ski.filters.gaussian(Idirac, sigma=self.gaussflt_sigma)
        return Ikernel

    def preprocess_image(self,
                         I: ArrayLike,
                         return_intermediates:bool = False) \
                            -> Union[NDArray, Tuple[NDArray]]:
        """Apply preprocessing on an image:

        1. Convert image to grayscale
        2. Invert the image
        3. Gaussian filtering

        Parameters
        ----------
        I : ArrayLike
            The input image
        return_intermediates: bool, optional
            If False, returns only the output image.
            If True, returns some intermediate images as well (useful
            for debugging): `return Ifilt, Iinv, Igray`
            Defaults to False.

        Returns
        -------
        Union[NDArray, Tuple[NDArray]]
            Either the output image, or a tuple with the output image
            and some intermediate images, according to `return_intermediates`.
        """

        # Convert to grayscale
        Igray = ski.color.rgb2gray(I)

        # Invert pixel values
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

    def plot_middleband_peaks(self, I: ArrayLike) -> Any:
        """Return an image with the midband peaks.

        Parameters
        ----------
        I : ArrayLike
            Input image

        Returns
        -------
        Any
            Matplotlib Figure object with the midband peaks. Useful for debugging.
        """

        # Find starting points from a middle band
        peaks, hist = self._find_starting_points(I)

        fig, ax = plt.subplots()
        ax.plot(hist)
        ax.plot(peaks, hist[peaks], "o")

        return fig

    def extract_ridges(self, 
                       I: ArrayLike,
                       stop_thresh: float = 0.5,
                       max_stride: int = 1) -> List[Ridge]:
        """Extract ridges from the image.

        Starting from the midband peaks, follow the ridge to the right
        and to the left, until the pixel value drops below `stop_thresh` * max.

        Parameters
        ----------
        I : ArrayLike
            The input image.
        max_stride: int, optional
            Search for the next max point in a vertical window [-max_stride, max_stride]
            around the current height.
            Defaults to 1.
        stop_thresh: float, optional
            Stop when the pixel value drops below `stop_thresh` * max.
            Defaults to 0.5
        stop_thresh: float, optional
            Stop when the pixel value drops below `stop_thresh` * max.

        Returns
        -------
        List[Ridge]
            List of Ridge objects.
        """

        # Find starting points from a middle band
        peaks, _ = self._find_starting_points(I)

        # Follow ridges and plot
        band_mid = int(I.shape[1] / 2)  # TODO: make in class
        ridges = [self._ridge_following(
            I, [peak, band_mid], max_stride=max_stride, stop_thresh=stop_thresh) for peak in peaks]

        return ridges

    @staticmethod
    def get_image_with_ridges(ridges: Iterable[Ridge],
                              I: Optional[ArrayLike] = None,
                              shape: Optional[Sequence] = None,
                              fg_color: Union[int, Tuple] = 0,
                              bg_color: Union[int, Tuple] = 1,
                              line: bool = False) -> NDArray:
        """Returns an image with overlaid ridges drawn on top.

        Parameters
        ----------
        ridges : Iterable[Ridge]
            Set of ridges
        I : Optional[ArrayLike], optional
            The underling image. If None, an image with size `shape` and
            color `bg_color` is generated.
            Defaults to None.
        shape : Optional[Tuple], optional
            (width, height) of image to generate, if I is None.
            By default None.
        fg_color : Union[int, Tuple], optional
            Color of segments, by default 0
        bg_color : Union[int, Tuple], optional
            Color of generated image, if I is None.
            Can be an int or a tuple of 3 ints (B,G,R)
            By default 1.
        line : bool, optional
            Plot only the points (False) or connect them (True).
            Defaults to False

        Returns
        -------
        NDArray
            Resulting image.
        """

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

