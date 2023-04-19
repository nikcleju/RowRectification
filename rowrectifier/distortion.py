import cv2
import math
import numpy as np
import sklearn
import sklearn.base

from typing import Tuple, List, Sequence, Iterator, Optional, Union
from numpy.typing import ArrayLike, NDArray

from itertools import pairwise
from scipy.interpolate import SmoothBivariateSpline

from .ridge import Ridge

class Distortion(sklearn.base.BaseEstimator):
    """Models the distortion between two images as matching points
    and ridges.
    """

    def __init__(self) -> None:
        """_summary_
        """
        super().__init__()

        # Parameters to be set in fit()
        self.ridges_ = None
        self.target_ridges_ = None
        self.target_shape_ = None

    def fit(self,
            ridges: Sequence[Ridge],
            target_shape:  Tuple,
            include_edges: bool = True) -> None:
        """Computes matching horizontal ridges for the provided ridges.

        Parameters
        ----------
        ridges : Sequence[Ridge]
            Sequence of Ridge objects describing paths which should be
            horizontal
        target_shape : Tuple
            Tuple with the output image size (width, height)
        include_edges : bool, optional
            Map the top and bottom edges of the initial image to the
            top and bottom edges of the final image, by default True.
            If True, adds them to `ridges`
        """

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
                self.ridges_.insert(
                    0, Ridge(ridges[0].x, [0] * len(ridges[0].y)))
                self.ridges_.append(
                    Ridge(ridges[-1].x,
                          [target_shape[0]-1] * len(ridges[-1].y)))

        # Make ridges horizontal
        self.target_ridges_ = [r.horizontalize(
                                        target_y='mean',
                                        x_start=0,
                                        x_end=target_shape[1]-1)
                               for r in self.ridges_]

    @property
    def matching_points(self) -> Iterator[Tuple]:
        """Return a generator which produces pairs of matching points,
        going through all the ridges.
        """

        for (src_ridge, tgt_ridge) in zip(self.ridges_, self.target_ridges_):
            for src_p, tgt_p in zip(src_ridge, tgt_ridge):
                yield (src_p, tgt_p)

    @property
    def matching_ridges(self) -> Iterator[Tuple]:
        """Return a generator which produces pairs of matching ridges.
        """

        for (src_ridge, tgt_ridge) in zip(self.ridges_, self.target_ridges_):
            yield (src_ridge, tgt_ridge)

    def get_image_with_shifts(self,
                              I: Optional[ArrayLike] = None,
                              shape: Optional[Tuple] = None,
                              fg_color: Union[int, Tuple] = 0,
                              bg_color: Union[int, Tuple] = 1,
                              thickness: int = 1) -> NDArray:
        """Returns an image with overlayed segments drawn between
        the matching original and target points.
        Useful for representing the dewarping.


        Parameters
        ----------
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
        thickness : int, optional
            Thickness of line segments, by default 1

        Returns
        -------
        NDArray
            Output image as a numpy array

        Raises
        ------
        NotImplementedError
            If I is None and shape is None.
        """

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

        # Draw the lines
        for ridge_src, ridge_dst in zip(self.ridges_, self.target_ridges_):
            for p_src, p_dst in zip(ridge_src, ridge_dst):
                Iout = cv2.line(Iout,
                                list(map(int,p_src)), list(map(int,p_dst)),
                                color=fg_color, thickness=thickness)
        return Iout


class DistortionPoly(Distortion):
    """Extends class `Distorsion` by decomposing an image into polygons
    and computing the rectified polygons.
    """

    def __init__(self, 
                 n_poly_sides: int = 3, 
                 tolerance: Optional[float] = None) -> None:
        """Constructor

        Parameters
        ----------
        n_poly_sides : int, optional
            Number of polygon sides (e.g. 3 = triangle).
            Can be either 3 or 4, defaults to 3.
        tolerance : Optional[float], optional
            If provided, skip polygons with area < tolerance.
            Currently only implemented for triangles.
            Defaults to None.
        """

        super().__init__()
        self.n_poly_sides = n_poly_sides
        self.tolerance = tolerance

    @property
    def matching_polys(self) -> Iterator[Tuple]:
        """Return a generator which produces pairs of matching polygons
        between input and target images.

        Yields
        ------
        Tuple
            (src, dst) where src is a polygon in the original image and
            dst is the matching polygon in the target image

        Raises
        ------
        ValueError
            Unsupported number of polygon sides.
        """

        if self.n_poly_sides == 4:
            # Produce 4-sided polygons
            for (src_ridge1, src_ridge2), (dst_ridge1, dst_ridge2) \
              in zip(pairwise(self.ridges_), pairwise(self.target_ridges_)):

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

            for (src_ridge1, src_ridge2), (dst_ridge1, dst_ridge2) \
              in zip(pairwise(self.ridges_), pairwise(self.target_ridges_)):

                points_src_all = np.concatenate(
                    [src_ridge1.to_numpy(), src_ridge2.to_numpy()])
                points_dst_all = np.concatenate(
                    [dst_ridge1.to_numpy(), dst_ridge2.to_numpy()])

                # Create triangles
                for indices in get_triangulation_indices(points_dst_all):

                    if self.tolerance is not None:
                        # Skip degenerate triangles (area too small)
                        p1,p2,p3 = tuple(points_src_all[indices])
                        if area_triangle(p1,p2,p3) < self.tolerance:
                            # TODO: replace print with logging
                            print(f"Degenerate src triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")
                        p1,p2,p3 = tuple(points_dst_all[indices])
                        if area_triangle(p1,p2,p3) < 1:
                            print(f"Degenerate dst triangle: {p1}, {p2}, {p3}, area={area_triangle(p1,p2,p3)}")

                    yield points_src_all[indices], points_dst_all[indices]
        else:
            raise ValueError(f"n_poly_sides must be 3 or 4, is {self.n_poly_sides}")

    def get_image_with_polys(self,
                             polys: str = 'src',
                             I: Optional[ArrayLike] = None,
                             shape: Optional[Tuple] = None,
                             fg_color: Union[int, Tuple] = 0,
                             bg_color: Union[int, Tuple] = 1,
                             thickness: int = 1) -> NDArray:
        """Returns an image with overlayed polygons.
        Useful for representing the polygonal decomposition of the image.

        Parameters
        ----------
        polys : str, optional
            Which polygons to draw: 'src' (source) or 'dst' (target)
            By default 'src'.
        I : Optional[ArrayLike], optional
            The underling image. If None, an image with size `shape` and
            color `bg_color` is generated.
            Defaults to None.
        shape : Optional[Tuple], optional
            (width, height) of image to generate, if I is None.
            By default None.
        fg_color : Union[int, Tuple], optional
            Color of lines, by default 0
        bg_color : Union[int, Tuple], optional
            Color of generated image, if I is None.
            Can be an int or a tuple of 3 ints (B,G,R)
            By default 1.
        thickness : int, optional
            Thickness of line segments, by default 1

        Returns
        -------
        NDArray
            Output image as a numpy array

        Raises
        ------
        NotImplementedError
            If I is None and shape is None.
        """

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

        # Get polygons to draw
        if polys == 'src':
            polys = [t[0] for t in self.matching_polys]
        elif polys == 'dst':
            polys = [t[1] for t in self.matching_polys]
        else:
            raise ValueError(f"polys '{polys}' not understood")

        # Prepare polys in OpenCV expected format and plot them.
        # Should be a list with arrays of size (4,1,2),
        # that's how bloddy OpenCV wants them.
        polys_cv2 = [np.array(poly, dtype=np.int).reshape((-1, 1, 2))
                     for poly in polys]
        Iout = cv2.polylines(
            Iout, polys_cv2, isClosed=True, color=fg_color, thickness=thickness)

        return Iout


class DistortionMap(Distortion):
    """Extends class `Distortion` by approximating the relation between
    mathching points with 2D splines.

    Uses a pair of 2D splines to approximate the functions:
        xin = f(xout, yout)
        yin = f(xout, yout)
    where (xout,yout) are the coordinates of points on the rectified image,
    and (xin,yin) on the original image.

    Then we can use these maps to get the source point for every point
    in the rectified image.

    """

    def __init__(self, delta: bool = False, margin: float = 200) -> None:
        """Constructor

        Parameters
        ----------
        delta : bool, optional
            Approximate the difference between target and original coordinates,
            instead of the raw target coordinates.
            By default False.
        margin: float, optional
            Extra margin around the image to define the bounding box
            (bbox) for spline approximations.
            By default equal to 200.

        """
        super().__init__()
        self.delta = delta
        self.margin = margin

    def fit_splines(self) -> None:
        """Compute the splines matching input points to output points.
        """

        # Concatenate x and y coordinates for all points
        # Target points = on the original image
        out_x = np.concatenate([r.x for r in self.ridges_])
        out_y = np.concatenate([r.y for r in self.ridges_])

        # Concatenate x and y coordinates for all output points
        # Input points = on the rectified image
        in_x = np.concatenate([r.x for r in self.target_ridges_])
        in_y = np.concatenate([r.y for r in self.target_ridges_])

        # Use delta or not
        target_x = out_x - in_x if self.delta else out_x
        target_y = out_y - in_y if self.delta else out_y

        # Bounding box for splines edges
        bbox=[min(in_x) - self.margin, max(in_x) + self.margin,
              min(in_y) - self.margin, max(in_y) + self.margin]

        # Find the splines
        self.spl_x_ = SmoothBivariateSpline(
            in_x, in_y, target_x, bbox=bbox, kx=3, ky=3)   # s = ...
        self.spl_y_ = SmoothBivariateSpline(
            in_x, in_y, target_y, bbox=bbox, kx=3, ky=3)   # s = ...

    def ev(self,
           x: ArrayLike,
           y: ArrayLike) -> Tuple[NDArray]:

        """Evaluate the splines at given points.

        Parameters
        ----------
        x: ArrayLike
            x coordinates of points
        y: ArrayLike
            y coordinates of points


        Returns
        -------
        Tuple[NDArray]
            A tuple (spl1(x,y), spl2(x,y)) with the values of the two splines.
        """

        if not hasattr(self, 'spl_x_') or not hasattr(self, 'spl_y_'):
            self.fit_splines()

        map_x = self.spl_x_(x, y).T
        map_y = self.spl_y_(x, y).T

        xmesh, ymesh = np.meshgrid(x, y)

        map_x = map_x + xmesh if self.delta else map_x
        map_y = map_y + ymesh if self.delta else map_y

        return map_x.astype(np.float32), map_y.astype(np.float32)

    def get_maps(self) -> Tuple[NDArray]:
        """Return the splines evaluated in all points of the target image.
        Can be afterwards used with OpenCV remap() to rectify the images.

        Returns
        -------
        Tuple[NDArray]
            A tuple with the values of the two splines in all points.
        """
        return self.ev(np.arange(self.target_shape_[1]),
                       np.arange(self.target_shape_[0]))

    def get_image_with_displacements(self,
                                     I: Optional[ArrayLike] = None,
                                     shape: Optional[Tuple] = None,
                                     grid_x: int = 20,
                                     grid_y: int = 20,
                                     fg_color: Union[int, Tuple] = 0,
                                     bg_color: Union[int, Tuple] = 1,
                                     thickness: int = 1) -> NDArray:
        """Returns an image with displacements for points on a grid.
        Useful for representing the spline mappings.

        Parameters
        ----------
        I : Optional[ArrayLike], optional
            The underling image. If None, an image with size `shape` and
            color `bg_color` is generated.
            Defaults to None.
        shape : Optional[Tuple], optional
            (width, height) of image to generate, if I is None.
            By default None.
        grid_x : int, optional
            Grid step in horizontal direction, by default 20
        grid_y : int, optional
            Grid step in vertical direction, by default 20
        fg_color : Union[int, Tuple], optional
            Color of lines, by default 0
        bg_color : Union[int, Tuple], optional
            Color of generated image, if I is None.
            Can be an int or a tuple of 3 ints (B,G,R)
            By default 1.
        thickness : int, optional
            Thickness of line segments, by default 1

        Returns
        -------
        NDArray
            Output image as a numpy array

        Raises
        ------
        NotImplementedError
            If I is None and shape is None.
        """

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

        # Compute grid
        eval_x = np.arange(0, I.shape[1], grid_x)
        eval_y = np.arange(0, I.shape[0], grid_y)

        # Evaluate splines
        displ_x, displ_y = self.ev(x=eval_x, y=eval_y)

        # Plot
        for ix, x in enumerate(eval_x):
            for iy, y in enumerate(eval_y):
                Iout = cv2.line(Iout, [int(x), int(y)],
                                [int(displ_x[iy, ix]), int(displ_y[iy, ix])],
                                color=fg_color,
                                thickness=thickness)

        return Iout

#====================
# Helper functions
#====================

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
    """Compute the area of a triangle
    """

    side1 = np.linalg.norm(p1-p2)
    side2 = np.linalg.norm(p2-p3)
    side3 = np.linalg.norm(p1-p3)

    s = (side1 + side2 + side3) / 2                         # Semi-perimeter
    area = math.sqrt((s*(s-side1)*(s-side2)*(s-side3)))     # Area
    return area
