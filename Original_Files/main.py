import field_distortion_read_in_files as fld
import FGCentroid as fg
import numpy as np
from astropy.io import fits
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as an
from celluloid import Camera as cam

def averageComparisons(directory,in1a,in1b,in2a,in2b,in3a,in3b,in4a,in4b):
    names = [in1a,in1b,in2a,in2b,in3a,in3b,in4a,in4b]
    avgs = [0,0,0,0,0,0,0,0]
    filesFits = []
    fitsNames = []
    setUsed = []
    currAvg = 0

    for files in os.listdir(directory):
        if files.endswith('.fits'):
            filesFits.append(files)

    for h in range(0,len(filesFits)):
        fitsNames.append(directory + '\\' + filesFits[h])

    for i in range(0,len(names)):
        for j in range(0,len(fitsNames)):
            if str(names[i]) in fitsNames[j]:
                setUsed.append(fitsNames[j])
    
        for k in range(0,len(setUsed)):
            currAvg = currAvg + np.sum(fits.open(setUsed[k])[1].data)/(1944*2592)
        if len(setUsed) != 0:
            currAvg = currAvg/len(setUsed)
        avgs[i] = currAvg

        setUsed = []
        currAvg = 0
    
    return names,avgs

whichFile = 3
"""
whichFile's Choices and their Functionalities

0: Bright Files Averaging
1: Fast Gaussian PSF Fitting
2: Dark Frame Off-Axis Averaging Over Full Image
3: Centroid Stability Analysis & Animation
4: Converts all .Raw files in a folder to .fits files for full frame images
5: Converts all .Raw files in a folder to .fits files for 2x2 binned images
99: Custom file viewing functionalities.
"""
if whichFile == 0:
    """
    User-defined variables
    """
    fldFile = r"C:\Users\antho\Videos\NG\Testing_4-4\Off-Axis_Images"
    fldBright = 'bright_negY_bNA_lightOn_edgeVis-'
    fldDark = 'dark_negP_bCF_lightOff_edgeOff-'
    
    """
    Program code, do not edit.
    """
    returns = fld.files_in(fldFile,fldBright,fldDark,4)
    print(returns)
    
elif whichFile == 1:
    """
    User-defined variables.
    """
    fgFile = r"C:\Users\antho\Videos\NG\Testing_2-10\PSFdata_2x2b\2x2_all\avg_lights_sub__0-p5.fits"
    fgArray = fits.getdata(fgFile)
    centerGuess = [168,616]             # ex. = [1119,1320]  <-backwards from 'SAOimage DS9' viewer
    spotSize = 5
    M1 = fgArray[(centerGuess[0]-spotSize):(centerGuess[0]+spotSize),(centerGuess[1]-spotSize):(centerGuess[1]+spotSize)]
    
    """
    Program code, do not edit.
    """
    centroid = fg.FGCentroid2(M=M1,pkRow=centerGuess[1],pkCol=centerGuess[0],Ncentr=(centerGuess[0]-centerGuess[0]),Method='Gaussian',SNRthresh=1)
    '''Takes into account corrected pixel positions for x and y (not z).
    in main.py:
    array = fits.getdata(filepath for avg file)
    M1 = array[row_start:row_end, col_start: col_end]
    
    pix_corr_x -> Make np.zeros((1944,2592))
    pix_corr_y -> Make np.zeros((1944,2592))
    pkRow -> row value for brightest pixel
    pkCol -> col value for brightest pixel
    Ncentr -> row_end - row_start
    Method -> "RotGaussian"
    SNRthresh, bright_x, bright_y -> 1 for all of them
    '''

    print(
        "\nFast Gaussian Centroiding Results \n" 
        "\n" 
        "Row of centroid:                        {:.4f} \n" 
        "Column of centroid:                     {:.4f} \n" 
        "Gaussian Amplitude:                     {:.4f} \n" 
        "Gaussian Sigma:                         {:.4f} \n" 
        "Gaussian Standard Deviation (x,y):     ({:.4f},{:.4f}) \n" 
        "Offset:                                 {:.4f} \n" 
        "Centroid Calculation (x,y):            ({:.4f},{:.4f}) \n" 
        "Centroid Calculation Adjusted (x,y):   ({:.4f},{:.4f}) \n" 
        "Brightest Pixel (x,y):                 ({:.4f},{:.4f}) \n" 
        .format(centroid['row'],centroid['col'],centroid['GauAmp'],centroid['GauSig'],centroid['GauSigx'],
                centroid['GauSigy'],centroid['offset'],centroid['x'],centroid['y'],centroid['x']+centerGuess[0]-spotSize,
                centroid['y']+centerGuess[1]-spotSize,centerGuess[0],centerGuess[1])
        )

    t = np.linspace(0.0,2.0*np.pi,100)
    x = (centroid['GauSigx'])*np.cos(t)
    y = (centroid['GauSigy'])*np.sin(t)

    plt.subplot(231)
    plt.imshow(np.reshape(np.round(centroid['GauImg'],1),(2*spotSize,2*spotSize)))
    plt.plot(x+centroid['x'],y+centroid['y'],'xkcd:bright teal',label=r'1$\sigma$')
    plt.plot(2*x+centroid['x'],2*y+centroid['y'],'xkcd:fluorescent green',label=r'2$\sigma$')
    plt.plot(3*x+centroid['x'],3*y+centroid['y'],'xkcd:orangered',label=r'3$\sigma$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.axis('equal')
    plt.legend()
    plt.title('Fast Gaussian Fitted PSF')
    plt.subplot(232)
    plt.imshow(np.reshape(np.round(centroid['GauImg'],1),(2*spotSize,2*spotSize)) - M1,aspect=1)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Original Minus Fitted')
    plt.subplot(233)
    plt.imshow(M1)
    plt.gca().invert_yaxis()
    plt.title('Original Image')
    plt.colorbar()
    plt.subplot(234)
    plt.imshow(np.reshape(np.round(centroid['GauImg'],1),(2*spotSize,2*spotSize)))
    plt.fill(x+centroid['x'],y+centroid['y'],'xkcd:bright teal',label=r'1$\sigma$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.legend()
    plt.title('1 Standard Deviation')
    plt.subplot(235)
    plt.imshow(np.reshape(np.round(centroid['GauImg'],1),(2*spotSize,2*spotSize)))
    plt.fill(2*x+centroid['x'],2*y+centroid['y'],'xkcd:fluorescent green',label=r'2$\sigma$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.legend()
    plt.title('2 Standard Deviations')
    plt.subplot(236)
    plt.imshow(np.reshape(np.round(centroid['GauImg'],1),(2*spotSize,2*spotSize)))
    plt.fill(3*x+centroid['x'],3*y+centroid['y'],'xkcd:orangered',label=r'3$\sigma$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.legend()
    plt.title('3 Standard Deviations')
    plt.show()
elif whichFile == 2:
    filePath = r'C:\Users\antho\Videos\NG\Testing_2-21'
    
    named,averages = averageComparisons(filePath,
                                        'posP_nb','posP_bPLA','posY_nb','posY_bPLA',
                                        'negP_nb','negP_bPLA','negY_nb','negY_bPLA')
    for i in range(0,len(averages)):
        print('Value for: %s is %f'%(named[i],averages[i]))
elif whichFile == 3:
    """
    User-defined variables
    """
    csaPath = r"C:\Users\antho\Videos\NG\Testing_2-10\PSFdata_ff\ff_all"
    imgBrightSpot = [1119,1320]
    csaSpotSize = 6
    saveAnimation = False
    animationName = 'ff_0-0_csa.gif'

    """
    Variables/code used by the program, do not edit. 
    """
    images = []
    subImages = []
    centroids = []
    csaSumX = 0
    csaSumY = 0
    csaAvgX = 0
    csaAvgY = 0

    csaFilePath = csaPath + "\\"
    for filename in os.listdir(csaFilePath):
        if not filename.endswith('.fits'):
            continue
        if 'bright_0-0_-02' in filename:
            fFile = fits.getdata(csaFilePath + filename)
            images.append(fFile)

    for j in range(0,len(images)):
        M2 = images[j][(imgBrightSpot[0]-csaSpotSize):(imgBrightSpot[0]+csaSpotSize),
                       (imgBrightSpot[1]-csaSpotSize):(imgBrightSpot[1]+csaSpotSize)]
        centroids.append(fg.FGCentroid2(M=M2,pkRow=imgBrightSpot[1],pkCol=imgBrightSpot[0],
                                        Ncentr=(imgBrightSpot[0]-imgBrightSpot[0]),Method='Gaussian',SNRthresh=1))
        subImages.append(M2)

    for w in range(0,len(centroids)):
        csaSumX = csaSumX + (centroids[w]['x']+imgBrightSpot[0]-csaSpotSize)
        csaSumY = csaSumY + (centroids[w]['y']+imgBrightSpot[1]-csaSpotSize)

    csaAvgX = csaSumX / len(centroids)
    csaAvgY = csaSumY / len(centroids)

    maxDist = float()
    minX = float()
    minY = float()
    maxX = float()
    maxY = float()
    for i in range(0,len(centroids)):
        dist = (np.sqrt(((centroids[i]['x']+imgBrightSpot[0]-csaSpotSize)-(csaAvgX))**2+((centroids[i]['y']+imgBrightSpot[1]-csaSpotSize)-(csaAvgY))**2))
        if dist > maxDist:
            maxDist = dist
            maxX = (centroids[i]['x']+imgBrightSpot[0]-csaSpotSize)
            maxY = (centroids[i]['y']+imgBrightSpot[1]-csaSpotSize)

    print(
        "Max Distance of: {:.4f}\n"
        "Max x-Distance of: {:.2f}\n"
        "Max y-Distance of: {:.2f}\n"
        .format(maxDist,maxX,maxY)
        )

    fig, ax = plt.subplots()
    ims = []
    camera = cam(fig)

    for p in range(0,len(subImages)):
        im = ax.imshow(subImages[p], animated=True)
        camera.snap()
        if p == 0:
            ax.imshow(subImages[p])
        ims.append([im])

    ani = an.ArtistAnimation(fig, ims, interval=200, repeat_delay=0)
    if saveAnimation == True:
        animation = camera.animate()
        animation.save(csaFilePath+animationName)
    plt.show()
elif whichFile == 4:
    """
    User-defined variable
    """
    direct = r"C:\Users\antho\Videos\NG\Testing_4-4\Off-Axis_Images"
    
    """
    Program code, do not edit.
    """
    for filename in os.listdir(direct):
        filepath = direct + "\\" + filename
        if not filepath.endswith(".Raw"):
            continue
        if os.path.isfile(filepath):
            raw_imarray = np.fromfile(filepath, dtype='uint16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
            fits_file = filepath.split('.')[0]+'.fits'

            """
            The following code segment is used to adjust the filenames of the output converted .fits files
            - Can be removed/changed as needed for different naming schemes 
            """
            hypSplit = fits_file.split('-')
            fits_file = hypSplit[0] + "-" + hypSplit[1] + "-" + hypSplit[2] + "-" + hypSplit[4]
            if len(fits_file.split("On")) == 1:
                direcSplt = fits_file.split("Images\\")
                fits_file = direcSplt[0] + "Images\\" + "dark_" + direcSplt[1] 
            else:
                direcSplt = fits_file.split("Images\\")
                fits_file = direcSplt[0] + "Images\\" + "bright_" + direcSplt[1] 
            """
            -- End of segment
            """

            hdu = fits.ImageHDU(reshaped_raw_imarray)
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim,hdu])
            hdu.writeto(fits_file, overwrite=True)
    print("\nAll .raw images converted to .fits\n")
elif whichFile == 5:
    """
    User-defined variable.
    """
    direct = r"C:\Users\antho\Videos\NG\Testing_4-4\Off-Axis_Images"
    
    """
    Program code, do not edit.
    """
    for filename in os.listdir(direct):
        filepath = direct + "\\" + filename
        if not filepath.endswith(".Raw"):
            continue
        if os.path.isfile(filepath):
            raw_imarray = np.fromfile(filepath, dtype='uint16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (int(1944/2 - 12),int(2592/2),2))
            fits_file = filepath.split('.')[0]+'.fits'
            
            """
            The following code segment is used to adjust the filenames of the output converted .fits files
            - Can be removed/changed as needed for different naming schemes 
            """
            hypSplit = fits_file.split('-')
            fits_file = hypSplit[0] + "-" + hypSplit[1] + "-" + hypSplit[2] + "-" + hypSplit[4]
            if len(fits_file.split("On")) == 1:
                direcSplt = fits_file.split("Images\\")
                fits_file = direcSplt[0] + "Images\\" + "dark_" + direcSplt[1] 
            else:
                direcSplt = fits_file.split("Images\\")
                fits_file = direcSplt[0] + "Images\\" + "bright_" + direcSplt[1] 
            """
            -- End of segment
            """

            hdu = fits.ImageHDU(reshaped_raw_imarray)
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim,hdu])
            hdu.writeto(fits_file, overwrite=True)
    print("\nAll .raw images converted to .fits\n")
elif whichFile == 99:
    folderPath = r"C:\Users\antho\Videos\NG\PSF_Characterization\p_0-0"
    fitsImgs = []
    for filename in os.listdir(folderPath):
        filepath = folderPath + '\\' + filename
        if not filepath.endswith('.fits'):
            continue
        if os.path.isfile(filepath):
            with fits.open(filepath) as hdul:
                image = hdul[1].data
                fitsImgs.append(image.astype(np.uint16))
    
    img = fitsImgs[0]
    vLines = False
    hLines = False
    centCircle = False
    topCircle = False
    leftCircle = False
    zoom = False
    zoomMid = [972,1296]
    zoomOffset = 200

    if zoom == True:
        plotTest = plt.imshow(img[zoomMid[0]-zoomOffset:zoomMid[0]+zoomOffset,zoomMid[1]-zoomOffset:zoomMid[1]+zoomOffset])
        plotTest = plt.clim(np.min(img),np.max(img))
        xVals = np.arange(zoomMid[0]-zoomOffset,zoomMid[0]+zoomOffset,50)
        yVals = np.arange(zoomMid[1]-zoomOffset,zoomMid[1]+zoomOffset,50)
        plotTest = plt.xticks(np.arange(0,zoomOffset*2,50),labels=xVals)
        plotTest = plt.yticks(np.arange(0,zoomOffset*2,50),labels=yVals)
    else:
        plotTest = plt.imshow(img)

    if centCircle == True:
        t = np.linspace(0.0,2.0*np.pi,100)
        x = ((2592/2)+200*np.cos(t))
        y = ((1944/2)+200*np.sin(t))
        plotTest = plt.plot(x,y,ls='dotted')

    if topCircle == True:
        t = np.linspace(0.0,2.0*np.pi,100)
        x = ((2592/2)+200*np.cos(t))
        y = ((1944)+200*np.sin(t))
        plotTest = plt.plot(x,y,ls='dotted')

    if leftCircle == True:
        t = np.linspace(0.0,2.0*np.pi,100)
        x = ((0)+200*np.cos(t))
        y = ((1944/2)+200*np.sin(t))
        plotTest = plt.plot(x,y,ls='dotted')

    if vLines == True:
        plt.axvline(1,color='yellow',label='p5-Pitch',lw=3)
        plt.axvline(259,color='limegreen',label='p4-Pitch')
        plt.axvline(518,color='teal',label='p3-Pitch')
        plt.axvline(777,color='mediumslateblue',label='p2-Pitch')
        plt.axvline(1036,color='indigo',label='p1-Pitch')
        plt.axvline(1296,color='black',label='0-Pitch')
        plt.axvline(1555,color='indigo',label='m1-Pitch')
        plt.axvline(1814,color='mediumslateblue',label='m2-Pitch')
        plt.axvline(2073,color='teal',label='m3-Pitch')
        plt.axvline(2332,color='limegreen',label='m4-Pitch')
        plt.axvline(2591,color='yellow',label='m5-Pitch',lw=3)
        plt.legend()

    if hLines == True:
        plt.axhline(1,color='yellow',label='Yaw-p5',lw=3)
        plt.axhline(194,color='limegreen',label='Yaw-p4')
        plt.axhline(388,color='teal',label='Yaw-p3')
        plt.axhline(583,color='mediumslateblue',label='Yaw-p2')
        plt.axhline(777,color='indigo',label='Yaw-p1')
        plt.axhline(972,color='black',label='Yaw-0')
        plt.axhline(1166,color='indigo',label='Yaw-m1')
        plt.axhline(1360,color='mediumslateblue',label='Yaw-m2')
        plt.axhline(1555,color='teal',label='Yaw-m3')
        plt.axhline(1749,color='limegreen',label='Yaw-m4')
        plt.axhline(1943,color='yellow',label='Yaw-m5',lw=3)
        plt.legend()

    if zoom != True:
        plotTest = plt.xlim(0,2592)
        plotTest = plt.ylim(0,1944)
    plt.show()
else:
    print('Please choose a valid input,',whichFile,'is not a possible choice.')


