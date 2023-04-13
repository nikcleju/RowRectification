# %%
import numpy as np
import scipy as sp
import scipy.interpolate, scipy.signal
import skimage as ski
import skimage.color, skimage.filters

import matplotlib.pyplot as plt
#plt.rcParams['figure.dpi'] = 300

from utils import ridge_following

I = plt.imread('Rows.png')
I = ski.color.rgb2gray(I)
# plt.figure()
# plt.imshow(I, cmap='gray')
# # plt.show()

# Invert
maxval = np.max(I)
Iinv = maxval - I
# plt.figure()
# plt.imshow(Iinv, cmap='gray')
# # plt.show()

# Gaussian filtering
If = ski.filters.gaussian(Iinv, sigma=[5, 50])
plt.figure()
plt.imshow(If, cmap='gray')
#plt.show()
#plt.imsave('If.png', If, cmap='gray')

# Histogram of a band in the middle part
band_x = int(I.shape[1] / 2)
band_halfwidth = 50
h = np.sum(If[:, band_x-band_halfwidth:band_x+band_halfwidth], axis=1)
# plt.figure()
# plt.plot(h)
# plt.show()

# Extract peaks
# Must do it automatically
#peaks = [99, 137, 181, 225, 275, 326, 375, 424, 473, 517]
#peaks = [40, 100, 150, 200, 250, 300, 352, 401, 450, 500]

peaks, _ = scipy.signal.find_peaks(h)
#plt.plot(h)
#plt.plot(peaks, h[peaks], "o")

If2 = np.copy(I)

# Follow ridges and plot
ridges = []
for peak in peaks:

    ridge = ridge_following(If, [peak,band_x], stop_thresh=0.5)

    # Mark on figure
    for c in range(If2.shape[1]):
        if ridge[c] != -1:
            If2[ridge[c],c] = 0

    ridges.append(ridge)

# plt.figure()
# plt.imshow(If2, cmap='gray')
# plt.imsave('If2.png', If2, cmap='gray')

# Smoothing
plt.figure()
for ridge in ridges:

    # Limit to part != -1
    x = np.array(range(I.shape[1]))
    x_actual = x[ridge != -1]    
    y_actual = ridge[ridge != -1]

    spl = sp.interpolate.UnivariateSpline(x_actual, y_actual, k=1, s=len(x)/2)
    plt.plot(x_actual, spl(x_actual))
plt.gca().invert_yaxis()
plt.show() 

# Smoothing 2D
plt.figure()
for ridge in ridges:

    # Limit to part != -1
    x = np.array(range(I.shape[1]))
    x_actual = x[ridge != -1]    
    y_actual = ridge[ridge != -1]

    spl = sp.interpolate.UnivariateSpline(x_actual, y_actual, k=1, s=len(x)/2)
    plt.plot(x_actual, spl(x_actual))
plt.gca().invert_yaxis()
plt.show() 


pass

