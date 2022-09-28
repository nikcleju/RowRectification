# %%
#import cv2
import numpy as np
import skimage as ski
import skimage.color, skimage.filters

import matplotlib
matplotlib.use('QtAgg')  # non-interactive
import matplotlib.pyplot as plt

# %%
I = plt.imread('Rows.png')
I = ski.color.rgb2gray(I)
#plt.imshow(I, cmap='gray')
#plt.show()

# %%
maxval = np.max(I)
I = maxval - I # invert
#plt.imshow(I, cmap='gray')
#plt.show()

# %%
# create the figure
#fig = plt.figure()
#ax = plt.axes(projection='3d')


# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:I.shape[0], 0:I.shape[1]]

# plot
#ax.plot_surface(xx, yy, I ,rstride=10, cstride=10, cmap=plt.cm.jet,
#                linewidth=0)

plt.show()

# %%
If = ski.filters.gaussian(I, sigma=[5, 50])
plt.imshow(If, cmap='gray')
plt.show()

# plot surface
fig = plt.figure()
ax = plt.axes(projection='3d')
xx, yy = np.mgrid[0:If.shape[0], 0:If.shape[1]]
ax.plot_surface(xx, yy, If ,rstride=10, cstride=10, cmap=plt.cm.jet, linewidth=0)
plt.show()
# %%


from skimage import data
from skimage import color
from skimage import filters
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt


def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image


image = color.rgb2gray(If)
cmap = plt.cm.gray

#kwargs = {'sigmas': [1], 'mode': 'reflect'}  # skimage latest 0.19
kwargs = {'sigmas': [10]}  # skimage version 0.16 has no 'mode'

fig, axes = plt.subplots(2, 5)
for i, black_ridges in enumerate([1, 0]):
    for j, func in enumerate([identity, meijering, sato, frangi, hessian]):
        kwargs['black_ridges'] = black_ridges
        result = func(image, **kwargs)
        axes[i, j].imshow(result, cmap=cmap, aspect='auto')
        if i == 0:
            axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                  'Sato\ntubeness', 'Frangi\nvesselness',
                                  'Hessian\nvesselness'][j])
        if j == 0:
            axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()

# %%
