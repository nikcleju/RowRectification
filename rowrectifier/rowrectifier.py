import numpy as np
import sklearn
import sklearn.base

from .ridge_extraction import RidgeExtractor
from .distortion import DistortionPoly, DistortionMap
from .rectification import RectifierPoly, RectifierMap

class RowRectifier(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, method, smoothing_method=None, n_segments=None,
                 n_poly_sides=None, k=None, s=None, delta=False) -> None:
        super().__init__()
        self.method = method
        self.smoothing_method = smoothing_method

        self.n_segments = n_segments
        self.n_poly_sides = n_poly_sides

        self.s = s

        self.delta = delta

        self.ir = None

    def fit(self, I, return_intermediates=False):

        # 1. Preprocess and extract ridges
        extr = RidgeExtractor()
        if return_intermediates:
            Ifilt, Iinv, Igray = extr.preprocess_image(I, return_intermediates=True)
            Ikernel = extr.get_image_kernel(
                (int(I.shape[0]/I.shape[1]*200), 200))
            Ipeaks = extr.plot_middleband_peaks(Ifilt)
        else:
            Ifilt = extr.preprocess_image(I, return_intermediates=False)

        ridges = extr.extract_ridges(Ifilt, stop_thresh=0.5)
        if return_intermediates:
            #Ifiltridges = extr.get_image_with_ridges(ridges, Ifilt, line=True)
            Ifiltridges = extr.get_image_with_ridges(ridges, I=None, line=True, bg_color=1)

        margin = 0

        # smooth ridges
        if self.smoothing_method == 'pwlf':
            ridges_smoothed = [ridge.smooth_pwlf(
                n=self.n_segments, endpoints=[0-margin, I.shape[1]-1+margin],
                clip_y=[0, I.shape[0]])
                    for ridge in ridges]
        elif self.smoothing_method == 'univariate_spline':

            # For poly method, include page endpoints in ridge,
            # so that the rectification extends to the page limits.
            # For map method, just use the matching points
            endpoints = [] if  self.method == 'map' else [0-margin, I.shape[1]-1+margin]
            ridges_smoothed = [ridge.smooth_univariatespline(
                k=1, s=self.s, endpoints=endpoints, clip_y=[0, I.shape[0]])
                    for ridge in ridges]

        elif self.smoothing_method is None:
            ridges_smoothed = ridges
        else:
                raise ValueError(f"Smoothing method {self.smoothing_method} not understood")

        # 2. Rectify
        if self.method == 'poly':
            self.ir = RectifierPoly(DistortionPoly(
                n_poly_sides=self.n_poly_sides,
                include_edges=True,
                tolerance=1,
                margin=0))
        elif self.method == 'map':
            self.ir = RectifierMap(DistortionMap(
                delta=self.delta,
                kx=3,
                ky=3,
                margin=200,
                include_edges=True))
        self.ir.fit(I, ridges=ridges_smoothed)

        if return_intermediates:
            return Ifiltridges, Ifilt, Iinv, Igray, Ikernel, Ipeaks

    def transform(self, I, return_intermediates=False):
        Iout = self.ir.transform(I)

        if return_intermediates:
            if self.method == 'poly':
                # Source image with polys
                Isrc_polys = self.ir.distortion.get_image_with_polys('src', I*255)

                # Source image with ridges and shifts
                Isrc_ridges = RidgeExtractor.get_image_with_ridges(self.ir.distortion.ridges_, I*255, fg_color=(0,0,255), line=True)
                #Isrc_ridges = self.ir.distortion.get_image_with_shifts(Isrc_ridges, fg_color=(255,0,0))

                # Rectified image with polys
                Idst_polys = self.ir.distortion.get_image_with_polys('dst', Iout*255)

                # Rectified image with ridges
                Idst_ridges = RidgeExtractor.get_image_with_ridges(self.ir.distortion.target_ridges_, 255*Iout, fg_color=(0,0,255), line=True)

                return Iout, Isrc_polys, Isrc_ridges, Idst_polys, Idst_ridges

            elif self.method == 'map':

                # Source image with ridges and shifts
                Isrc_ridges = RidgeExtractor.get_image_with_ridges(self.ir.distortion.ridges_, I*255, fg_color=(0,0,255), line=True)
                Isrc_ridges = self.ir.distortion.get_image_with_shifts(Isrc_ridges, fg_color=(255,0,0))

                # Rectified image with ridges
                Idst_ridges = RidgeExtractor.get_image_with_ridges(self.ir.distortion.target_ridges_, 255*Iout, fg_color=(0,0,255), line=True)

                # Rectified image with ridges
                Idst_ridges = RidgeExtractor.get_image_with_ridges(self.ir.distortion.target_ridges_, 255*Iout, fg_color=(0,0,255), line=True)

                # Displacement
                Idispl = self.ir.distortion.get_image_with_displacements(I=255*Iout, fg_color=(0,0,255),
                                                                         grid_x=40, grid_y=80)

                # Maps
                map_x, map_y = self.ir.distortion.get_maps(delta=True)


                return Iout, Isrc_ridges, Idst_ridges, Idispl, map_x, map_y
            else:
                raise ValueError(f"Method {self.method} unknown")
        else:
            return Iout
