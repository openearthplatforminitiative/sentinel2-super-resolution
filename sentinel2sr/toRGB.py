import math
import numpy as np

# Contrast enhance / highlight compress
maxR = 3.0 # max reflectance
midR = 0.13
sat = 1.2
gamma = 1.8
scalefac = 10000

gOff = 0.01
gOffPow = math.pow(gOff, gamma)
gOffRange = math.pow(1 + gOff, gamma) - gOffPow


def toRGB(image):
    rgbLin = satEnh(sAdj(image[0]/scalefac), sAdj(image[1]/scalefac), sAdj(image[2]/scalefac))
    rgb = np.array([sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2])]) * 255
    return np.floor(rgb)


def sAdj(a):
    return adjGamma(adj(a, midR, 1, maxR))


def adjGamma(b):
    return (np.pow((b + gOff), gamma) - gOffPow)/gOffRange


# Saturation enhancement
def satEnh(r, g, b):
    avgS = (r + g + b) / 3.0 * (1 - sat)
    return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]


def clip(s):
    s[s < 0] = 0
    s[s > 1] = 1
    return s


# contrast enhancement with highlight compression
def adj(a, tx, ty, maxC):
    ar = clip(a / maxC)
    return ar * (ar * (tx/maxC + ty - 1) - ty) / (ar * (2 * tx/maxC - 1) - tx/maxC)


def sRGB(c):
    mask = (c <= 0.0031308)
    np.putmask(c, mask, 12.92 * c)
    np.putmask(c, np.invert(mask), 1.055 * np.pow(c, 0.41666666666) - 0.055)
    return c
