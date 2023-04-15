"""Defines the Ridge class and associates
"""

import numpy as np
import pwlf
import scipy as sp

from itertools import pairwise
from typing import Sequence, Generator, List


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
    def from_points(cls, points):
        return cls([p[0] for p in points], [p[1] for p in points])

    def __iter__(self) -> Generator:
        """Returns an iterator to iterate through the pairs (x,y) pairs on the
        ridge.

        Returns
        -------
        _type_
            Returns `zip(self.x, self.y)`
        """
        return zip(self.x, self.y)

    @property
    def segment_lengths(self) -> List:
        """Returns a list with the length of the segments on the ridge.
        A segment is defined by two consecutive points on the ridge.

        Returns
        -------
        list(float)
            List of segment lengths on the ridge
        """
        return [np.linalg.norm(np.array(p2)-np.array(p1)) for p1,p2 in pairwise(self)]

    @property
    def length(self) -> float:
        """Return total length of the ridge (i.e. sum of all segment 
        lengths)

        Returns
        -------
        float
            Total length of the ridge
        """
        return sum(self.segment_lengths)

    @property
    def _get_points(self):
        return [pair for pair in zip(self.x, self.y)]

    # Methods
    def normalize(self):

        # Compute cumulative sum and normalize
        cumsum = np.cumsum(self.segment_lengths) / self.length

        # Prepend 0
        cumsum = np.insert(cumsum, 0, 0) 

        return Ridge(cumsum, self.y)

    # Smoothing
    def smooth_univariatespline(self, k=1, s=None, endpoints=[], predict_each_point=False, clip_y=None):

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

    def smooth_pwlf(self, n=4, endpoints=[], predict_each_point=False, clip_y=None):

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

    def to_numpy(self):
        #return np.ndarray([self.x, self.y]).T
        return np.vstack((self.x, self.y)).T

    # TODO: from dict etc

