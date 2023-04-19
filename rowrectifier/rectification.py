import cv2
import numpy as np

from .distortion import Distortion, crop_to_poly

class Rectifier:

    def __init__(self, distortion):
        self.distortion = distortion

    def fit(self, I, ridges):
        self.distortion.fit(ridges, target_shape=I.shape)

    def transform():
        raise NotImplementedError("Must be overridden in derivated classes")

class RectifierPoly(Rectifier):
    def transform(self, I):
        return rectify_image(I, poly_pairs=self.distortion.matching_polys)

class RectifierMap(Rectifier):
    def transform(self, I):
        map_x, map_y = self.distortion.get_maps()
        return cv2.remap(I, map_x, map_y, cv2.INTER_LINEAR)


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

def rectify_image(I, polys_src=None, polys_dst=None, poly_pairs=None):

    # Prepare new matrix
    Iout = np.zeros_like(I)

    if polys_src is not None and polys_dst is not None and poly_pairs is None:
        poly_pairs = zip(polys_src, polys_dst)
    elif polys_src is None and polys_dst is None and poly_pairs is not None:
        pass
    else:
        raise ValueError(f"Parameter combination not understood")

    # HACK
    crop_poly = False

    for poly_src, poly_dst in poly_pairs:

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
                               (dst_img_cropped.shape[1], dst_img_cropped.shape[0]))

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
