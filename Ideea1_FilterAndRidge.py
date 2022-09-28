# %%
#import cv2
import numpy as np
import skimage as ski
import skimage.color, skimage.filters

import matplotlib
matplotlib.use('Qt5Agg')  # non-interactive
import matplotlib.pyplot as plt

from utils import ridge_following

I = plt.imread('Rows.png')
I = ski.color.rgb2gray(I)
# plt.imshow(I, cmap='gray')
# plt.show()

# Invert
maxval = np.max(I)
Iinv = maxval - I
# plt.imshow(I, cmap='gray')
# plt.show()

# Gaussian filtering
If = ski.filters.gaussian(Iinv, sigma=[5, 50])
plt.imshow(If, cmap='gray')
plt.show()

# Histogram of a band in the middle part
mid_width = int(I.shape[1] / 2)
band_halfwidth = 50
h = np.sum(If[:,mid_width - band_halfwidth : mid_width + band_halfwidth], axis=1)
plt.plot(h)
plt.show()

# Extract peaks
# Must do it automatically
#peaks = [99, 137, 181, 225, 275, 326, 375, 424, 473, 517]
peaks = [40, 100, 150, 200, 250, 300, 352, 401, 450, 500]

If2 = np.copy(I)

# Follow ridges and plot
for peak in peaks:

    ridge = ridge_following(If, [peak,mid_width], stop_thresh=0.5)

    # Mark on figure
    for c in range(If2.shape[1]):
        if ridge[c] != -1:
            If2[ridge[c],c] = 0

plt.imshow(If2, cmap='gray')
plt.show() 

