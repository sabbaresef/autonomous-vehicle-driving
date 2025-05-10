import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from io import BytesIO
from PIL import Image
from PIL import Image as PILImage
from skimage.filters import gaussian

#####################################################################################################################
#####################################################################################################################
# Helpful functions
def random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal(mean,maxval/2,(n,1)).reshape((n,))
    gauss = np.clip(gauss, mean-maxval, mean+maxval)
    if positive:
        return np.abs(gauss)
    else:
        return gauss



def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
    Args:
        img_height: Height of image cutout mask will be applied to.
        img_width: Width of image cutout mask will be applied to.
        num_channels: Number of channels in the image.
        size: Size of the zeros mask.
    Returns:
        A mask of shape `img_height` x `img_width` with all ones except for a
        square of zeros of shape `size` x `size`. This mask is meant to be
        elementwise multiplied with the original image. Additionally returns
        the `upper_coord` and `lower_coord` which specify where the cutout mask
        will be applied.
    """
    #assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                    min(img_width, width_loc + size // 2))

    mask_height = lower_coord[0] - upper_coord[0] 
    mask_width = lower_coord[1] - upper_coord[1]

    if mask_height <= 0:
        mask_height = 1
    
    if mask_width <= 0:
        mask_width = 1

    mask = np.ones((img_height, img_width, num_channels), dtype = np.float32)
    zeros = np.zeros((mask_height, mask_width, num_channels), dtype=np.float32)
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (zeros)

    return mask, upper_coord, lower_coord


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()
    
#####################################################################################################################
#####################################################################################################################



def jpeg_compression(x):
    """Apply a jpeg compression to the image."""
    
    c = [25, 18, 15, 10, 7, 5, 3, 1]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    if len(x.shape)==3 and x.shape[2]==1:
        x = np.squeeze(x,2)
    
    x = x * 255.
    x = Image.fromarray(x.astype(np.uint8))

    output = BytesIO()

    x.save(output, 'JPEG', quality=c)

    x = PILImage.open(output)
    x = np.array(x)

    if len(x.shape)==2:
        x = np.expand_dims(x,2)

    return (x.astype(np.float32)) / 255.


def random_brightness_contrast(img):
    """Brightness and contrast"""
    
    a = random_normal_crop(1, 0.5, mean=1)[0]
    b = random_normal_crop(1, 48)[0]
    img = img * 255.
    img = (img-128.0)*a + 128.0 + b
    img = np.clip(img, 0, 255)
    img = (img.astype(np.float32)) / 255.

    return img


def cutout(img):
    """Apply cutout with mask of shape `min_shape` x `min_shape` to `img`.
    This operation applies a `min_shape`x`min_shape` mask of zeros to a random location
    within `img`.
    Args:
        img: Numpy image that cutout will be applied to.
    Returns:
        A numpy tensor that is the result of applying the cutout mask to `img`.
    """
    
    img_height, img_width, num_channels = (img.shape[0], img.shape[1], img.shape[2])
    min_shape = min(img_height, img_width)

    size = np.random.randint(0, min_shape * np.random.uniform(0.1, 0.6))

    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)

    output = img * mask
    return output.astype(np.float32)


def gaussian_noise(x):
    """Apply gaussian noise to the image."""
    
    c = [0.1, 0.2, .04, .08, 0.12, 0.16, 0.20]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    x = np.array(x)

    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1).astype(np.float32)


def impulse_noise(x):
    """Apply impulse noise to the image."""
    
    c = [.005, .01, .03, .05, .07, 0.13, 0.19]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    x = sk.util.random_noise(np.array(x), mode='s&p', amount=c)
    return np.clip(x, 0, 1).astype(np.float32)


def gaussian_blur(x):
    """Apply gaussian blur to the image."""
    
    c = [.1, .3, .5, 1, 1.8, 2.2, 2.6, 3.4, 4.0]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    x = gaussian(np.array(x), sigma=c, multichannel=True)
    return np.clip(x, 0, 1).astype(np.float32)


def defocus_blur(x):
    """Apply defocus blur to the image."""
    
    c = [(1, 0.05), (1.3, 0.07), (1.5, 0.1), (2, 0.2), (2, 0.3), (2.5, 0.4), (3, 0.4)]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    c = tuple([item/48*x.shape[0] for item in c])
    
    x = np.array(x)
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(x.shape[2]):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
    
    return np.clip(channels, 0, 1).astype(np.float32)


def fog(x):
    """Add fog to the image."""
    
    c = [(1., 2.5), (1.2, 2.2), (1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    x = np.array(x)
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1).astype(np.float32)


def saturate(x):
    """Changing the saturation of the image"""
    
    c = [(0.05, 0), (0.1, 0), (0.1, 0.1), (0.3, 0), (2, 0), (5, 0.1), (10, 0.2), (15, 0.2)]
    severity = np.random.randint(0, len(c) - 1)
    c = c[severity]

    x = np.array(x)
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1).astype(np.float32)