import cv2
import matplotlib.pyplot as plt
import numpy as np
import pwlf
import scipy as sp
import scipy.interpolate
import scipy.signal
import skimage as ski
import skimage.color, skimage.filters
import sklearn
import sklearn.base

from itertools import pairwise
from dataclasses import dataclass

#========================
# Ridges
#========================

class Ridge:
    
    def __init__(self, x: list[int], y:list[int]) -> None:
        self.x = x
        self.y = y
    
    def __iter__(self):
        return zip(self.x, self.y)
    #     self._gen = zip(self.x, self.y)   # need to find a better way
    #     return self
    
    # def __next__(self):
    #     return next(self._gen)

    # Getter functions for properties
    def _get_segment_lengths(self):
        return [np.linalg.norm(np.array(p2)-np.array(p1)) for p1,p2 in pairwise(self)]

    def _get_length(self):
        return sum(self.segment_lengths)

    # Properties
    length = property(_get_length)
    segment_lengths = property(_get_segment_lengths)

    # Methods
    def normalize(self):

        # Compute cumulative sum and normalize
        cumsum = np.cumsum(self.segment_lengths) / self.length

        # Prepend 0
        cumsum = np.insert(cumsum, 0, 0)  

        return Ridge(cumsum, self.y)

    # Smoothing
    def smooth_univariatespline(self, k=1, s=None):

        s = len(self.x)/2 if s is None else s

        spl = sp.interpolate.UnivariateSpline(self.x, self.y, k=k, s=s)
        new_y = [spl(x) for x in self.x]

        return Ridge(self.x, new_y)


    # Smoothing
    def smooth_pwlf(self, n=4, endpoints=[], predict_each_point=False):

        # initialize piecewise linear fit with your x and y data
        my_pwlf = pwlf.PiecewiseLinFit(self.x, self.y)

        # fit the data for four line segments
        knots = my_pwlf.fitfast(n, pop=3)  # TODO: default pop=2, check this

        # Add extremes to knots
        if endpoints:
            knots = np.insert(knots,0,endpoints[0])
            knots = np.append(knots,endpoints[1])
        
        if predict_each_point:
            y = my_pwlf.predict(np.arange(knots[0], knots[-1]))
        else:
            y = my_pwlf.predict(knots)
        
        return Ridge(knots,y)

    # TODO: from dict etc


#========================
# Preprocessing 
#========================

class RidgeExtractor:
    def __init__(self) -> None:
        self.gaussflt_sigma = [5,50]

    def get_kernel_image(self, shape):
        Idirac = np.zeros(shape)
        Idirac[shape[0]//2,shape[1]//2] = 1
        Ikernel = ski.filters.gaussian(Idirac, sigma=self.gaussflt_sigma)
        #plt.imshow(Ikernel, cmap='gray')    
        return Ikernel

    def preprocess_image(self, I):

        Igray = ski.color.rgb2gray(I)
        #plt.imshow(I, cmap='gray')

        # Invert image
        maxval = np.max(Igray)
        Iinv = maxval - Igray
        #plt.imshow(Iinv, cmap='gray')

        # Gaussian filtering
        Ifilt = ski.filters.gaussian(Iinv, sigma=self.gaussflt_sigma)

        return Ifilt, Iinv, Igray

    def _find_starting_points(self, I):
        # Find starting points from a middle band
        band_mid = int(I.shape[1] / 2)
        band_halfwidth = 50
        hist = np.sum(I[:, band_mid-band_halfwidth:band_mid+band_halfwidth], axis=1)
        peaks, _ = scipy.signal.find_peaks(hist)

        return peaks, hist

    def get_middleband_peaks_image(self, I):
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

    def plot_ridges_on_image(self, ridges, I=None, shape=None, fg_color=0, bg_color=1, line=False):

        # Prepare background image
        if I is not None:
            Iout = np.copy(I)
        if I is None and shape is not None:
            Iout = np.zeros(shape) + bg_color
        elif I is None and shape is None:
            max_x = max([max([x for x in ridge.x]) for ridge in ridges])
            max_y = max([max([y for y in ridge.y]) for ridge in ridges])
            Iout = np.zeros((max_y, max_x))  + bg_color      

        # Plot ridges on image, 
        for ridge in ridges:
            if line:
                for p1, p2 in pairwise(ridge):
                    Iout = cv2.line(Iout, list(map(int,p1)), list(map(int,p2)), color=(0,0,0))
            else:
                for x,y in ridge:
                    Iout[int(y), int(x)] = fg_color
        # for c in range(I.shape[1]):
        #     if ridge[c] != -1:
        #         Imask[ridge[c],c] = 0

        # plt.figure()
        # plt.imshow(Imask, cmap='gray')        
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
            
            ridge[i] = ridge[i-1] - max_stride + np.argmax(I[ridge[i-1] - max_stride : ridge[i-1] + max_stride + 1, i])

            # Update max value
            max_value = max(max_value, I[ridge[i],i])

            # Check termination
            if I[ridge[i],i] < stop_thresh * max_value:
                ridge[i] = -1
                break

        # Follow ridge to the left of start_point
        max_value = I[start_row, start_col]
        for i in range(start_col-1, -1, -1):

            ridge[i] = ridge[i+1] - max_stride + np.argmax(I[ridge[i+1] - max_stride : ridge[i+1] + max_stride + 1, i])

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


#========================
# Distortion models 
#========================

class DistortionPoly(sklearn.base.BaseEstimator):
    def __init__(self, n_segments) -> None:
        super().__init__()
        self.n_segments = n_segments

        self.polys_src_ = None
        self.polys_dst_ = None

    def fit(self, ridges, target_shape):

        # Add top and bottom margins as ridges
        ridges.insert(0, Ridge(ridges[0].x, [0] * len(ridges[0].y)))
        ridges.append(Ridge(ridges[-1].x, [target_shape[0]] * len(ridges[0].y)))

        # Poligonify original ridges in the image
        self.polys_src_ = []
        for ridge1, ridge2 in pairwise(ridges):
            for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
                self.polys_src_.append([topleft, topright, botright, botleft])

        # Compute target ridges
        # x values are normalized to total length, y values are all equal to first y value of original ridge (horizontal)
        target_length = target_shape[1]
        target_ridges = [Ridge( ridge.normalize().x * target_length, [ridge.y[0]] * len(ridge.y)) for ridge in ridges]

        # Poligonify target ridges
        self.polys_dst_ = []
        for ridge1, ridge2 in pairwise(target_ridges):
            for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
                self.polys_dst_.append([topleft, topright, botright, botleft])

    def __iter__(self): 
        if not hasattr(self, 'polys_src_') or not hasattr(self, 'polys_dst_'):
            raise ValueError('Polygons not computed, have you run fit()?')
        
        return zip(self.polys_src_, self.polys_dst_)

    def plot_polys_on_image(self, polys='src', I=None, shape=None, fg_color=0, bg_color=1, thickness=1):

        # Prepare background image
        if I is not None:
            Iout = np.copy(I)
        if I is None and shape is not None:
            Iout = np.zeros(shape) + bg_color
        elif I is None and shape is None:
            #max_x = max([max([x for x in ridge.x]) for ridge in ridges])
            #max_y = max([max([y for y in ridge.y]) for ridge in ridges])
            #Iout = np.zeros((max_y, max_x))  + bg_color      
            raise NotImplementedError("Not implemented yet, TODO")

        # Plot ridges on image, 
        if polys == 'src':
            polys = self.polys_src_
        elif polys == 'dst':
            polys = self.polys_dst_
        else:
            raise ValueError(f"polys '{polys}' not understood")

        # Prepare polys in OpenCV expected format
        # list with arrays of size (4,1,2), that's how OpenCV bloody wants it
        polys_cv2 = [np.array(poly, dtype=np.int).reshape((-1, 1, 2)) for poly in polys ] 
        Iout = cv2.polylines(Iout, polys_cv2, isClosed=False, color=fg_color, thickness=thickness)
  
        return Iout        

    

#========================
# Image rectification    
#========================

class ImageRectifierPoly:

    def __init__(self, distortion_est: DistortionPoly) -> None:
        self.distortion_est_ = distortion_est

    def fit(self, I, ridges):
        self.distortion_est_.fit(ridges, target_shape=I.shape)

        # DEBUG
        Isrc = self.distortion_est_.plot_polys_on_image('src', I*255)
        cv2.imwrite('a.png', Isrc)
        Idst = self.distortion_est_.plot_polys_on_image('dst', I*255)
        cv2.imwrite('b.png', Idst)
        
    def transform(self, I):
        return rectify_image(I, self.distortion_est_.polys_src_, self.distortion_est_.polys_dst_)


# Rectify a 3 or 4-side polygon
def rectify_poly(I, poly_src, poly_dst):
    
    #assert len(poly_src) == len(poly_dst), "poly_src and poly_dst have different len"
    if len(poly_src) == 4 and len(poly_dst) == 4:
        # use perspective transformation for 4-side polygons
        persp_mat = cv2.getPerspectiveTransform(poly_src, poly_dst, cv2.DECOMP_LU)
        Ipersp = cv2.warpPerspective(I, persp_mat, I.shape[1::-1], flags=cv2.INTER_LINEAR)
    elif len(poly_src) == 3 and len(poly_dst) == 3:
        raise NotImplementedError("Transformation for triangles not implemented yet")
    else:
        raise NotImplementedError("Transformation for this polygon length not implemented")
    
    return Ipersp    

def rectify_image(I, polys_src, polys_dst):

    # Prepare new matrix
    Iout = np.zeros_like(I)

    for poly_src, poly_dst in zip(polys_src, polys_dst):
        Iwarped = rectify_poly(I, np.array(poly_src, dtype = "float32"), np.array(poly_dst, dtype = "float32")) 
            
        # Create mask for the triangle we want to transform
        # mask = np.zeros(dst_img_cropped.shape, dtype = np.uint8)
        # cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);

        # # Delete all existing pixels at given mask
        # dst_img_cropped*=1-mask
        # # Add new pixels to masked area
        # dst_img_cropped+=dst_img_warped*mask    

        mask = np.zeros_like(Iwarped, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(poly_dst), (1.0, 1.0, 1.0), 16, 0);

        # Delete all existing pixels at given mask
        Iout*=1-mask
        # Add new pixels to masked area
        Iout+=Iwarped*mask    

    return Iout


#========================
# Top-level row rectification class
#========================

class RowRectifierPwlf(sklearn.base.BaseEstimator, 
                       sklearn.base.TransformerMixin):

    def __init__(self, n_segments) -> None:
        super().__init__()
        self.n_segments = n_segments

        self.ir_ = None

    def fit(self, I):

        # 1. Preprocess and extract ridges
        extr = RidgeExtractor()
        Ifilt, Iinv, Igray = extr.preprocess_image(I)
        ridges = extr.extract_ridges(Ifilt)

        # smooth ridges
        ridges_pwlf = [ridge.smooth_pwlf(n=self.n_segments, endpoints=[0, I.shape[1]-1]) for ridge in ridges]

        # 2. Rectify 
        self.ir_ = ImageRectifierPoly(DistortionPoly(n_segments=4))
        self.ir_.fit(I, ridges=ridges_pwlf)

    def transform(self, I):
        return self.ir_.transform(I)

