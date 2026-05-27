from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

filename = r"C:\Users\antho\Videos\NG\Testing_1-31\image-70us.fits"

def SNR(img,brightX,brightY,darkX,darkY,sizes):
    startChange = np.floor_divide(sizes-1,2)
    brightStart = [brightX-startChange,brightY-startChange]
    darkStart = [darkX-startChange,darkY-startChange]
    signal = 0
    noise = 0

    for i in range(sizes):
        for j in range(sizes):
            # print("Signal[%i,%i]: "%(i,j),signal)
            signal += img[brightStart[0]+i,brightStart[1]+j]
            # print("Signal Added[%i,%i]: "%(i,j),img[brightStart[0]+i,brightStart[1]+j])
            j += 1
        i += 1

    for k in range(sizes):
        for l in range(sizes):
            # print("Noise[%i,%i]: "%(k,l),noise)
            noise += img[darkStart[0]+k,darkStart[1]+l]
            # print("Signal Added[%i,%i]: "%(k,l),img[darkStart[0]+k,darkStart[1]+l])
            k += 1
        l += 1

    grossTotals = [signal,noise]
    ratio = "The SNR for '%s' is - %.3f:%.0f"%(filename[38:49],grossTotals[0]/noise,grossTotals[1]/noise)
    return ratio,grossTotals

with fits.open(filename) as hdul:
    # hdul.info()
    print()
    image = hdul[1].data
    # print(image.shape)
    # print(image[781,1339])
    print(SNR(image,781,1339,794,1326,3))

    # plt.imshow(image)
    # plt.show()