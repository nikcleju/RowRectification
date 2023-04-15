import cv2
import math
import numpy as np
import sklearn
import sklearn.base

from itertools import pairwise
from scipy.interpolate import SmoothBivariateSpline

from .ridge import Ridge

class Distortion(sklearn.base.BaseEstimator):
    def __init__(self, n_poly_sides=3, delta=False) -> None:
        super().__init__()
        self.n_poly_sides = n_poly_sides

        self.ridges_ = None
        self.target_ridges_ = None
        self.target_shape_ = None
        self.delta = delta

    def fit(self, ridges, target_shape, include_edges=True):

        self.ridges_ = ridges
        self.target_shape_ = target_shape

        # Add top and bottom margins as ridges
        if include_edges:
            # Append only corners?
            append_only_corners = False  # Not working only with corners
            if append_only_corners:
                self.ridges_.insert(0, Ridge.from_points(
                    [[0,0],
                    [target_shape[1]-1, 0]]))
                self.ridges_.append(Ridge.from_points(
                    [[0, target_shape[0]-1],
                    [target_shape[1]-1, target_shape[0]-1]]))
            else:
                self.ridges_.insert(0, Ridge(ridges[0].x, [0] * len(ridges[0].y)))
                self.ridges_.append(Ridge(ridges[-1].x, [target_shape[0]-1] * len(ridges[-1].y)))

        # Compute target ridges:
        #   x values are normalized to total length, 
        #   y values are all equal (horizontal)
        self.target_ridges_ = [r.horizontalize(target_y='mean', 
                                               x_start=0, 
                                               x_end=target_shape[1]-1) 
                                               for r in self.ridges_]

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


class DistortionPoly(Distortion):

    def __init__(self, n_poly_sides=3) -> None:
        super().__init__()
        self.n_poly_sides = n_poly_sides


    @property
    def poly_pairs(self):
        """Return a generator which produces pairs of matching polygons
        between input and target images

        Yields
        ------
        tuple
            (src, dst) where src is a polygon in the original image and 
            dst is the matching polygon in the target image

        Raises
        ------
        ValueError
            Parameter errors
        """
        
        if self.n_poly_sides == 4:
            # Produce 4-sided polygons
            for (src_ridge1, src_ridge2), (dst_ridge1, dst_ridge2) in zip(pairwise(self.ridges_), pairwise(self.target_ridges_)):
                for (src_topleft, src_topright), (src_botleft, src_botright), \
                    (dst_topleft, dst_topright), (dst_botleft, dst_botright) \
                    in zip(pairwise(src_ridge1), pairwise(src_ridge2), \
                           pairwise(dst_ridge1), pairwise(dst_ridge2)):
                    yield ([src_topleft, src_topright, \
                            src_botright, src_botleft], \
                           [dst_topleft, dst_topright, \
                            dst_botright, dst_botleft])

        elif self.n_poly_sides == 3:
            # Get triangular mesh between two ridges only, with OpenCV
            for (src_ridge1, src_ridge2), (dst_ridge1, dst_ridge2) in zip(pairwise(self.ridges_), pairwise(self.target_ridges_)):
                points_src_all = np.concatenate([src_ridge1.to_numpy(), src_ridge2.to_numpy()])
                points_dst_all = np.concatenate([dst_ridge1.to_numpy(), dst_ridge2.to_numpy()])
                # Create triangles
                for indices in get_triangulation_indices(points_dst_all):
                    # # Get triangles from indices
                    # self.polys_src_.append( points_src_all[indices] )
                    # self.polys_dst_.append( points_dst_all[indices] )

                    p1,p2,p3 = tuple(points_src_all[indices])
                    if area_triangle(p1,p2,p3) < 1:
                        print(f"Degenerate src triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")
                    p1,p2,p3 = tuple(points_dst_all[indices])
                    if area_triangle(p1,p2,p3) < 1:
                        print(f"Degenerate dst triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")

                    yield points_src_all[indices], points_dst_all[indices]
        else:
            raise ValueError(f"n_poly_sides must be 3 or 4, is {self.n_poly_sides}")

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

        # Plot ridges on image
        if polys == 'src':
            polys = [t[0] for t in self.poly_pairs]
        elif polys == 'dst':
            polys = [t[1] for t in self.poly_pairs]
        else:
            raise ValueError(f"polys '{polys}' not understood")

        # Prepare polys in OpenCV expected format
        # list with arrays of size (4,1,2), that's how OpenCV bloody wants it
        polys_cv2 = [np.array(poly, dtype=np.int).reshape((-1, 1, 2)) for poly in polys ]
        Iout = cv2.polylines(Iout, polys_cv2, isClosed=True, color=fg_color, thickness=thickness)

        return Iout


class DistortionMap(Distortion):

    def __init__(self, delta=False) -> None:
        super().__init__()
        self.delta = delta

    def fit_splines(self):

        # Concatenate x and y coordinates for all points
        # self.ridges_ = target data for spline approximation
        # self.target_ridges_ = input data for spline approximation

        out_x = np.concatenate([r.x for r in self.ridges_])
        out_y = np.concatenate([r.y for r in self.ridges_])

        # Concatenate x and y coordinates for all output points
        in_x = np.concatenate([r.x for r in self.target_ridges_])
        in_y = np.concatenate([r.y for r in self.target_ridges_])

        # Target values of splines
        target_x = out_x - in_x if self.delta else out_x
        target_y = out_y - in_y if self.delta else out_y

        # spl_x, map_x = self._bivariate_spline_map(in_x, in_y, target_x)
        # spl_y, map_y = self._bivariate_spline_map(in_x, in_y, target_y)

        bbox=[min(in_x) - 200, max(in_x) + 200, 
              min(in_y) - 200, max(in_y) + 200]

        self.spl_x_ = SmoothBivariateSpline(in_x, in_y, target_x, 
                                      bbox=bbox, kx=3, ky=3)   # s = ...
        self.spl_y_ = SmoothBivariateSpline(in_x, in_y, target_y, 
                                      bbox=bbox, kx=3, ky=3)   # s = ...

        return self.spl_x_, self.spl_y_

    def get_maps(self):
        return self.ev(np.arange(self.target_shape_[1]), 
                       np.arange(self.target_shape_[0]))

    def ev(self, x, y):

        if not hasattr(self, 'spl_x_') or not hasattr(self, 'spl_y_'):
            self.fit_splines()

        map_x = self.spl_x_(x, y).T
        map_y = self.spl_y_(x, y).T

        xmesh, ymesh = np.meshgrid(x, y)

        map_x = map_x + xmesh if self.delta else map_x
        map_y = map_y + ymesh if self.delta else map_y

        return map_x.astype(np.float32), map_y.astype(np.float32)

    def get_image_with_displacements(self, I=None, shape=None, grid_x=20, grid_y=20, fg_color=0, bg_color=1, thickness=1):

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

        eval_x = np.arange(0, I.shape[1], grid_x)
        eval_y = np.arange(0, I.shape[0], grid_y)

        displ_x, displ_y = self.ev(x=eval_x, y=eval_y)

        for ix, x in enumerate(eval_x):
            for iy, y in enumerate(eval_y):
                Iout = cv2.line(Iout, [int(x), int(y)], 
                                [int(displ_x[iy, ix]), int(displ_y[iy, ix])], 
                                color=fg_color,
                                thickness=thickness)

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
