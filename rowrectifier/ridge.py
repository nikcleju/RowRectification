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

"""Defines the Ridge class and associates
"""

# For type hint a return value of same class
from __future__ import annotations

import numpy as np
import pwlf
import scipy as sp

from itertools import pairwise
from typing import Tuple, List, Sequence, Iterable, \
    Iterator, Generator, Optional, Union, Any
from numpy.typing import ArrayLike, NDArray

class Ridge:
    """A Ridge is a path in an image which follows a row.
    """

    def __init__(self, x: Sequence, y: Sequence) -> None:
        """Constructs a Ridge object from x and y data.

        The x and y coordinates are defined as:

        - origin (0,0) is top left corner
        - x defines the horizontal displacement (e.g. column number)
        - y defines the vertical displacement (e.g. row number)

        Parameters
        ----------
        x : Sequence
            The x coordinates of the points on the ridge
        y : Sequence
            The y coordinates of the points on the ridge
        """

        self.x = x
        self.y = y

    @classmethod
    def from_points(cls, points: Sequence) -> Ridge:
        """Generate a ridge from a sequence of points (x,y)

        Parameters
        ----------
        points : Sequence
            A sequence of (x,y) representing consecutive points.

        Returns
        -------
        Ridge
            The constructed Ridge object
        """
        return cls([p[0] for p in points], [p[1] for p in points])

    def __iter__(self) -> Generator:
        """Returns an iterator to iterate through the points (x,y) on the
        ridge.

        Returns
        -------
        Generator
            A generator `zip(self.x, self.y)`
        """
        return zip(self.x, self.y)

    @property
    def segment_lengths(self) -> List[float]:
        """Returns a list with the length of the segments on the ridge.
        A segment is defined by two consecutive points on the ridge.

        Assumes the points on the ridge are in the correct order.

        Returns
        -------
        list(float)
            List of segment lengths on the ridge
        """
        return [np.linalg.norm(np.array(p2)-np.array(p1)) for p1,p2 in pairwise(self)]

    @property
    def length(self) -> float:
        """Return total length of the ridge (i.e. sum of all segments
        lengths).

        Assumes the points on the ridge are in the correct order.

        Returns
        -------
        float
            Total length of the ridge
        """
        return sum(self.segment_lengths)

    @property
    def _get_points(self) -> List[tuple]:
        """Returns a list of point coordinates (x,y)

        Returns
        -------
        List[tuple]
            List of points (x,y)
        """
        return [pair for pair in zip(self.x, self.y)]

    # Methods
    def normalize(self) -> Ridge:
        """Normalize x values on a ridge to the total length.
        The x values will be in the range [0,1].

        Assumes the points on the ridge are in the correct order.

        Returns
        -------
        Ridge
            A new Ridge object with normalized x values.
        """

        # Compute cumulative sum and normalize
        cumsum = np.cumsum(self.segment_lengths) / self.length

        # Prepend 0
        cumsum = np.insert(cumsum, 0, 0)

        return Ridge(cumsum, self.y)

    # Smoothing
    def smooth_univariatespline(self,
                                k: int = 1,
                                s: Optional[float] = None,
                                endpoints: Sequence = [],
                                predict_each_point: bool = False,
                                clip_y: Optional[Sequence] = None) -> Ridge:
        """Smooth the ridge with a univariate spline, using
        scipy.interpolate.UnivariateSpline.

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline, defaults to 1.
            See `scipy.interpolate.UnivariateSpline` for details.
        s : Optional[float], optional
            Positive smoothing factor used to choose the number of knots,
            defaults to None.
            See `scipy.interpolate.UnivariateSpline` for details.
        endpoints : Sequence, optional
            Either empty [], or with two endpoints to be included at the
            beginning and the end of the ridge.
            Defaults to empty [].
        predict_each_point : bool, optional
            If True, the returned Ridge containes all the points
            in the range (first x, last x).
            If False, the returned Ridge contains only the points x
            in the original Ridge.
            Defaults to False.
        clip_y : Optional[Sequence], optional
            A sequence (y_min, y_max) to clip the y values to
            If None, no clipping is done.
            Defaults to None.

        Returns
        -------
        Ridge
            A new Ridge object where the y values are approximated with
            a smoothing spline.
        """

        s = len(self.x)/2 if s is None else s
        spl = sp.interpolate.UnivariateSpline(self.x, self.y, k=k, s=s)

        # Reference x  = the knots
        new_x = spl.get_knots()

        # Add extremes to x, only if they are not already in x (distance > epsilon)
        if endpoints:
            if np.abs(endpoints[0] - new_x[0]) > 1e-3:
                new_x = np.insert(new_x,0,endpoints[0])
            if np.abs(endpoints[1] - new_x[-1]) > 1e-3:
                new_x = np.append(new_x,endpoints[1])

        # Find y values
        if predict_each_point:
            new_y = spl(np.arange(new_x[0], new_x[-1]))
        else:
            new_y = spl(new_x)

        # Limit values in y to image height
        # Values may exceed image in endpoints, due to extrapolation
        if clip_y is not None:
            new_y = np.clip(new_y, clip_y[0], clip_y[1])

        if len(self.x) != len(self.y):
            print("ERROR")
            print(self.x)
            print(self.y)

        # Make x and y into np arrays
        new_x = np.array(new_x)
        new_y = np.array(new_y)

        return Ridge(new_x, new_y)

    def smooth_pwlf(self,
                    n:int = 4,
                    endpoints: Sequence = [],
                    predict_each_point: bool = False,
                    clip_y: Optional[Sequence] = None) -> Ridge:

        """Approximate the ridge with a fixed number of segments, using
        the `pwlf` package.

        Parameters
        ----------
        n : int
            Number of segments. Defaults to 4.
            See `pwlf` package for details.
        endpoints : Sequence, optional
            Either empty [], or with two endpoints to be included at the
            beginning and the end of the ridge.
            Defaults to empty [].
        predict_each_point : bool, optional
            If True, the returned Ridge containes all the points
            in the range (first x, last x).
            If False, the returned Ridge contains only the points x
            in the original Ridge.
            Defaults to False.
        clip_y : Optional[Sequence], optional
            A sequence (y_min, y_max) to clip the y values to
            If None, no clipping is done.
            Defaults to None.

        Returns
        -------
        Ridge
            A new Ridge object where the y values are approximated with
            a smoothing spline.
        """

        # initialize piecewise linear fit with your x and y data
        my_pwlf = pwlf.PiecewiseLinFit(self.x, self.y)

        # fit the data for n line segments
        knots = my_pwlf.fitfast(n, pop=3)  # TODO: default pop=2, check this

        # Add extremes to knots
        if endpoints:
            knots = np.insert(knots,0,endpoints[0])
            knots = np.append(knots,endpoints[1])

        # Find y values
        if predict_each_point:
            new_y = my_pwlf.predict(np.arange(knots[0], knots[-1]))
        else:
            new_y = my_pwlf.predict(knots)

        # Limit values in y to image height
        # Values may exceed image in endpoints, due to extrapolation
        if clip_y is not None:
            new_y = np.clip(new_y, clip_y[0], clip_y[1])

        return Ridge(knots,new_y)

    def horizontalize(self, target_y='mean', x_start=None, x_end=None):

        # What is the target y in the rectified image?
        if target_y == 'first':
            target_y = self.y[0]
        elif target_y == 'last':
            target_y = self.y[-1]
        elif target_y == 'mean':
            target_y = np.mean(self.y)
        else:
            raise ValueError(f"{target_y} not defined")

        x_start = self.x[0] if x_start is None else x_start
        x_end = self.x[-1] if x_end is None else x_end

        n = self.normalize()
        new_x = x_start + n.x * (x_end - x_start)
        new_y = [target_y] * len(self.y)

        return Ridge(new_x, new_y)

    def to_numpy(self) -> NDArray:
        """Convert to numpy arrat

        Returns
        -------
        NDArray
            A numpy array 2D with two rows, first row contains the x values,
            second row the y values of the points.
        """
        #return np.ndarray([self.x, self.y]).T
        return np.vstack((self.x, self.y)).T

    # TODO: from dict etc

