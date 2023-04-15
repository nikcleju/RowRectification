import cv2
import math
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


    # Smoothing
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

    def to_numpy(self):
        #return np.ndarray([self.x, self.y]).T

        if len(self.x) != len(self.y):
            print("ERROR")
            print(self.x)
            print(self.y)

        return np.vstack((self.x, self.y)).T

    # TODO: from dict etc


#========================
# Preprocessing 
#========================

class RidgeExtractor:
    def __init__(self) -> None:
        self.gaussflt_sigma = [5,50]

    def get_image_kernel(self, shape):
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

        # Plot ridges on image, 
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
    def __init__(self, n_segments, n_poly_sides=4) -> None:
        super().__init__()
        self.n_segments  = n_segments
        self.n_polysides = n_poly_sides

        self.ridges_ = None
        self.target_ridges_ = None
        self.polys_src_ = None
        self.polys_dst_ = None

    def fit(self, ridges, target_shape):

        # Add top and bottom margins as ridges
        self.ridges_ = ridges
        self.ridges_.insert(0, Ridge(ridges[0].x, [0] * len(ridges[0].y)))
        self.ridges_.append(Ridge(ridges[-1].x, [target_shape[0]-1] * len(ridges[-1].y)))

        # What is the target y in the rectified image?
        conf_target_ridge_y = 'mean' # or 'first
        if conf_target_ridge_y == 'first':
            target_ridge_y = [ridge.y[0] for ridge in self.ridges_]
        elif conf_target_ridge_y == 'last':
            target_ridge_y = [ridge.y[-1] for ridge in self.ridges_]
        elif conf_target_ridge_y == 'mean':
            target_ridge_y = [np.mean(ridge.y) for ridge in self.ridges_]

        # Compute target ridges
        # x values are normalized to total length, y values are all equal (horizontal) (ridge.y[0], or mean y)
        target_length = target_shape[1]
        self.target_ridges_ = [Ridge( ridge.normalize().x * (target_length-1), [tgt_y] * len(ridge.y)) for ridge, tgt_y in zip(self.ridges_, target_ridge_y)]

        # Poligonify original ridges in the image as well as the target ridges
        self.polys_src_ = []
        self.polys_dst_ = []

        if self.n_polysides == 4:
            # Create 4-sided polygons
            for ridge1, ridge2 in pairwise(self.ridges_):
                for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
                    self.polys_src_.append([topleft, topright, botright, botleft])

            for ridge1, ridge2 in pairwise(self.target_ridges_):
                for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
                    self.polys_dst_.append([topleft, topright, botright, botleft])
        
        elif self.n_polysides == 3:

            # # Get triangular mesh with OpenCV
            # points_src_all = np.concatenate([r.to_numpy() for r in        self.ridges_], axis=0)
            # points_dst_all = np.concatenate([r.to_numpy() for r in self.target_ridges_], axis=0)
            # # Create triangles
            # for indices in get_triangulation_indices(points_src_all):
            #     # Get triangles from indices
            #     src_triangle = points_src_all[indices]
            #     dst_triangle = points_dst_all[indices]
            #     self.polys_src_.append(src_triangle)
            #     self.polys_dst_.append(dst_triangle)

            # # Create two 3-sided polygons between two ridges
            # for ridge1, ridge2 in pairwise(self.ridges_):
            #     for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
            #         self.polys_src_.append([topleft, topright, botright])
            #         self.polys_src_.append([topleft, botleft, botright])
            #
            # for ridge1, ridge2 in pairwise(self.target_ridges_):
            #     for (topleft, topright), (botleft, botright) in zip(pairwise(ridge1), pairwise(ridge2)):
            #         self.polys_dst_.append([topleft, topright, botright])
            #         self.polys_dst_.append([topleft, botleft, botright])

            # Get triangular mesh between two ridges only, with OpenCV
            for (src_ridge1, src_ridge2), (dst_ridge1, dst_ridge2) in zip(pairwise(self.ridges_), pairwise(self.target_ridges_)):
                points_src_all = np.concatenate([src_ridge1.to_numpy(), src_ridge2.to_numpy()])
                points_dst_all = np.concatenate([dst_ridge1.to_numpy(), dst_ridge2.to_numpy()])
                # Create triangles
                for indices in get_triangulation_indices(points_dst_all):
                    # Get triangles from indices
                    self.polys_src_.append( points_src_all[indices] )
                    self.polys_dst_.append( points_dst_all[indices] )

                    p1,p2,p3 = tuple(points_src_all[indices])
                    if area_triangle(p1,p2,p3) < 1:
                        print(f"Degenerate src triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")
                    p1,p2,p3 = tuple(points_dst_all[indices])                        
                    if area_triangle(p1,p2,p3) < 1:
                        print(f"Degenerate dst triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")

        else:
            raise ValueError(f"n_poly_sides must be 3 or 4, is {self.n_poly_sides}")

    def __iter__(self): 
        if not hasattr(self, 'polys_src_') or not hasattr(self, 'polys_dst_'):
            raise ValueError('Polygons not computed, have you run fit()?')
        
        return zip(self.polys_src_, self.polys_dst_)

    def get_image_with_polys(self, polys='src', I=None, shape=None, fg_color=0, bg_color=1, thickness=1):

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
        Iout = cv2.polylines(Iout, polys_cv2, isClosed=True, color=fg_color, thickness=thickness)
  
        return Iout        

    def get_image_with_shifts(self, I=None, shape=None, fg_color=0, bg_color=1, thickness=1):

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

        for ridge_src, ridge_dst in zip(self.ridges_, self.target_ridges_):
            for p_src, p_dst in zip(ridge_src, ridge_dst):
                Iout = cv2.line(Iout, list(map(int,p_src)), list(map(int,p_dst)), color=fg_color)

        return Iout    

def get_triangulation_indices(points):
    """Get indices triples for every triangle
    """
    # Make points int
    points = np.int32(points)

    # Bounding rectangle
    # HACK: +1 otherwise final points raise error with insert()
    bounding_rect = (*points.min(axis=0), *points.max(axis=0)+1)
    #bounding_rect = (*points.min(axis=0), *points.max(axis=0))

    # Triangulate all points
    subdiv = cv2.Subdiv2D(bounding_rect)
    for p in points:
        subdiv.insert([int(p[0]), int(p[1])])

    # Iterate over all triangles
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        # Get index of all points
        yield [(points==point).all(axis=1).nonzero()[0][0] for point in [(x1,y1), (x2,y2), (x3,y3)]]


def crop_to_poly(img, poly):
    """Crop image to a polygon
    """
    # Get bounding rectangle
    bounding_rect = cv2.boundingRect(np.array(poly, dtype=int))

    # Crop image to bounding box
    img_cropped = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                      bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # Move triangle to coordinates in cropped image
    poly_cropped = [(point[0]-bounding_rect[0], point[1]-bounding_rect[1]) for point in poly]
    return poly_cropped, img_cropped

def area_triangle(p1, p2, p3):

    side1 = np.linalg.norm(p1-p2)
    side2 = np.linalg.norm(p2-p3)
    side3 = np.linalg.norm(p1-p3)
    
    s = (side1 + side2 + side3) / 2                         # Semi-perimeter
    area = math.sqrt((s*(s-side1)*(s-side2)*(s-side3)))     # Area
    return area

#========================
# Image rectification    
#========================

class ImageRectifierPoly:

    def __init__(self, distortion_est: DistortionPoly) -> None:
        self.distortion_est_ = distortion_est

    def fit(self, I, ridges):
        self.distortion_est_.fit(ridges, target_shape=I.shape)

    def transform(self, I):
        return rectify_image(I, self.distortion_est_.polys_src_, self.distortion_est_.polys_dst_)


# Rectify a 3 or 4-side polygon
def rectify_poly(I, poly_src, poly_dst, Iout_shape=None):
    
    if Iout_shape is None:
        Iout_shape = I.shape[1::-1]

    #assert len(poly_src) == len(poly_dst), "poly_src and poly_dst have different len"
    if len(poly_src) == 4 and len(poly_dst) == 4:
        # use perspective transformation for 4-side polygons
        persp_mat = cv2.getPerspectiveTransform(poly_src, poly_dst, cv2.DECOMP_LU)
        Irectif = cv2.warpPerspective(I, persp_mat, Iout_shape, flags=cv2.INTER_LINEAR)
    elif len(poly_src) == 3 and len(poly_dst) == 3:
        #raise NotImplementedError("Transformation for triangles not implemented yet")

        affine_mat = cv2.getAffineTransform(poly_src, poly_dst)
        Irectif = cv2.warpAffine(I, affine_mat, Iout_shape, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )        
    else:
        raise NotImplementedError("Transformation for this polygon length not implemented")
    
    return Irectif    

def rectify_image(I, polys_src, polys_dst):

    # Prepare new matrix
    Iout = np.zeros_like(I)

    # HACK
    crop_poly = False

    for poly_src, poly_dst in zip(polys_src, polys_dst):

        if crop_poly:
            # Crop the image around the polygon, for efficiency
            poly_src_cropped, src_img_cropped = crop_to_poly(I, poly_src)
            poly_dst_cropped, dst_img_cropped = crop_to_poly(Iout, poly_dst)
        else:
            poly_src_cropped, src_img_cropped = poly_src, I
            poly_dst_cropped, dst_img_cropped = poly_dst, Iout
            
        Iwarped = rectify_poly(src_img_cropped, 
                               np.array(poly_src_cropped, dtype = "float32"), 
                               np.array(poly_dst_cropped, dtype = "float32"), 
                               (dst_img_cropped.shape[1], dst_img_cropped.shape[0])
                               ) 
            
        # # Calculate transfrom to warp from old image to new
        # transform = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))
        # # Warp image
        # dst_img_warped = cv2.warpAffine(src_img_cropped, transform, (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        # # Create mask for the triangle we want to transform
        # mask = np.zeros(dst_img_cropped.shape, dtype = np.uint8)
        # cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);
        # # Delete all existing pixels at given mask
        # dst_img_cropped*=1-mask
        # # Add new pixels to masked area
        # dst_img_cropped+=dst_img_warped*mask

        mask = np.zeros_like(dst_img_cropped, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(poly_dst_cropped), (1.0, 1.0, 1.0), 16, 0);

        # Delete all existing pixels at given mask
        dst_img_cropped*=1-mask
        # Add new pixels to masked area
        dst_img_cropped+=Iwarped*mask    

    return Iout


#========================
# Top-level row rectification class
#========================

class RowRectifier(sklearn.base.BaseEstimator, 
                   sklearn.base.TransformerMixin):

    def __init__(self, smoothing_method, n_segments=None, n_poly_sides=None, k=None, s=None) -> None:
        super().__init__()
        self.smoothing_method = smoothing_method

        self.n_segments   = n_segments
        self.n_poly_sides = n_poly_sides

        self.s = s

        self.ir_ = None

    def fit(self, I):

        # 1. Preprocess and extract ridges
        extr = RidgeExtractor()
        Ifilt, Iinv, Igray = extr.preprocess_image(I)
        ridges = extr.extract_ridges(Ifilt)

        # smooth ridges
        if self.smoothing_method == 'pwlf':
            ridges_smoothed = [ridge.smooth_pwlf(n=self.n_segments, endpoints=[0, I.shape[1]-1], clip_y=[0, I.shape[0]]) for ridge in ridges]
        elif self.smoothing_method == 'univariate_spline':
            # k must be for polygonal rectification
            ridges_smoothed = [ridge.smooth_univariatespline(k=1, s=self.s, endpoints=[0, I.shape[1]-1], clip_y=[0, I.shape[0]]) for ridge in ridges]

        # 2. Rectify 
        self.ir_ = ImageRectifierPoly(DistortionPoly(n_segments=self.n_segments, n_poly_sides=self.n_poly_sides))
        self.ir_.fit(I, ridges=ridges_smoothed)

    def transform(self, I, return_intermediates=False):
        Iout = self.ir_.transform(I)

        if return_intermediates:
            # Source image with polys
            Isrc_polys = self.ir_.distortion_est_.get_image_with_polys('src', I*255)

            # Source image with ridges and shifts
            Isrc_ridges = RidgeExtractor.get_image_with_ridges(self.ir_.distortion_est_.ridges_, I*255, fg_color=(0,0,255), line=True)
            Isrc_ridges = self.ir_.distortion_est_.get_image_with_shifts(Isrc_ridges, fg_color=(255,0,0))
            
            # Rectified image with polys
            Idst_polys = self.ir_.distortion_est_.get_image_with_polys('dst', Iout*255)

            # Rectified image with ridges
            Idst_ridges = RidgeExtractor.get_image_with_ridges(self.ir_.distortion_est_.target_ridges_, 255*Iout, fg_color=(0,0,255), line=True)

            return Iout, Isrc_polys, Isrc_ridges, Idst_polys, Idst_ridges
        
        else:
            return Iout
