import numpy as np
from plotter import *
import scipy.ndimage.filters as scfilters
import scipy.ndimage as ndimage
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.collections
import tifffile
# from skimage import exposure
from intIndexCalc import *
# from numba import jit, njit
from saverAndLoader import *
from plotter import *
from multiproc import *
from FRETcalc import *
# from skimage import data, img_as_float, filters
import time
from pathlib import Path
import os
import traceback
# import winsound
class Object(object):
    pass


def analyzeMaster(d):
    startTime = time.time()
    # Source file directory #GUI
    # d['directorySource'] = 'W:\\IZKF2\\data\\MICROSCOPE-REPOSITORY\\TIRFM\\SYNC\\Camera Images\\2022\\20220621-HJ-b and b_complement\\' #GUI
    # d['directorySource'] = 'C:\\Feiz\\20211207\\anew\\'
    # d['directorySource'] = 'C:\\Feiz\\+testFRET\\'
    # d['directoryAlt'] = 'C:/Feiz/20220621/c/' #GUI
    # d['directoryAlt'] = d['directorySource']
    # fileName = 'c.tif' #GUI
    uManagerFile = 0
    testfile = 0
    #### user parameters
    # d['recordingMode'] = 'alex' # simple, prism, alex #GUI
    # d['FPS'] = 5 #frame per second #GUI
    # d['markEventFrame'] = 0 #if there is an event: event frame to marke in plots#GUI
    # d['integMin'] = 150 # minimum value for spot integration to be accepted Green: 200-500, Red:130-330#GUI
    # d['integMax'] = 1000 # maximum value for spot integration to be accepted#GUI
    # d['integMin2'] = 150 # minimum value for spot integration to be accepted Red:130-330, Green: 200-500#GUI
    # d['integMax2'] = 1000#GUI
    # d['sigmaImg'] = 1 # sigma for image blurring #GUI
    # d['sigmaBkgnd'] = 20 # sigma to make a background (def = 40) #GUI
    # d['corrIlluPatterns'] = 1 #correct for the difference between illuminations of ch.1 and ch.2#GUI
    # d['realBkGnd'] = 0 #calculate background by first removing spots #GUI
    # d['integralSpotShape1'] = 'circle' # circle,fit,average #GUI
    # d['intR'] = 4 # radius for intensity integration #GUI
    # d['blinkDuration'] = 10 #frames to wait for blinking (min = 1)#GUI
    # d['seekR'] = 8 #seeking radius for maximums#GUI
    # d['FFS'] = 10 # First Frames Sampling for finding peaks#GUI
    # d['movingAvgN'] = 1 #moving average number for plotting #GUI
    d['removeFirstDarkFrames'] = 0
    d['darkFrameMeanValueThreshold'] = 170
    # d['alexSpotDist'] = [-10, 0] #[y , x] distance between alex spots pair, acceptor to donor, y and x from top left #GUI
    # d['QE'] = 0.78 #GUI
    # d['ADfactor'] = 0.457 # e/count #GUI
    ###########################################
    ###########################################
    ### manage files to read
    d['channelNo'] = 0
    # fileName = d['fname']
    # folderList = [d['directorySource']]
    # if uManagerFile:
    #     folderList=[x[0] for x in os.walk(d['directorySource']+'/')]
    # for fileNo, folderName in enumerate(folderList):
    try:
        #make directory
        # folderName = folderName[folderName.rfind('/'):]
        # if uManagerFile:
        #     fileName = folderName + folderName+ '_MMStack_Pos0.ome.tif'
        d['fname'] = d['fnameExt'][0:d['fnameExt'].rfind('.')]
        d['directorySave'] = d['directorySaveParent']+ '/' + d['fname'] + '/'
        Path(d['directorySave']).mkdir(parents=True, exist_ok=True)
        # d['fname'] = fileName
        # if testfile:
        #     d['directorySource'] = 'C:/Feiz/+testFRET/'
        #     d['directorySave'] = d['directorySource']
        #     d['fnameExt'] = 'a.tiff'
        ####################
        d['index2photon'] = d['ADfactor']/d['QE']
        d['figNo'] = 1
        (imgMatrix, d) = imgImporter(d)
        ### reduce matrix size for test:
        # imgMatrix = imgMatrix[:1650, :, 0:15]
        d['imgX'] = imgMatrix.shape[1]
        d['imgY'] = imgMatrix.shape[0]
        d['totFrames'] = imgMatrix.shape[2]
        if d['recordingMode'] == 'singleChannelDoubleDye':
            d['channelNo'] = 1
        if d['recordingMode'] == 'alex':
            d['channelNo'] = 1
            imgMatrix2 = imgMatrix[:, :, 1::2] #separate 2nd channel
            imgMatrix = imgMatrix[:, :, ::2] #separate 1st channel
            d2 = d.copy() # dummy dict for 2nd channel to use instead of main channel in other functions
            index2photon2 = d['ADfactor']/d['QE2']
            d['totFrames'] = imgMatrix.shape[2]
            d['totFrames2'] = imgMatrix2.shape[2]
            d['imgSample2'] = imgMatrix2[:, :, 0:d['FFS2']]
            d['firstImgs2'] = index2photon2 * np.sum(d  ['imgSample2']/d['FFS2'], 2)
            d['firstImg2'] = index2photon2 * imgMatrix2[:, :, 0]
            d['firstImgSTD2'] = np.std(d['firstImg2'])
            # d['firstImg2'] = filters.gaussian(d['firstImg2'], sigma=d['sigmaImg']) - filters.gaussian(d['firstImg2'], sigma=d['sigmaBkgnd'])
            d2['totFrames'] = d['totFrames2']
            d2['imgSample'] = d['imgSample2']
            d2['firstImgs'] = d['firstImgs2']
            d2['firstImg'] = d['firstImg2']
            d2['firstImgSTD'] = d['firstImgSTD2']
        d['imgSample'] = imgMatrix[:, :, 0:d['FFS']] # first few frames to extract spots
        d['firstImgs'] = d['index2photon'] * np.sum(d['imgSample']/d['FFS'], 2)
        d['firstImg']  = d['index2photon'] * imgMatrix[:, :, 0]
        d['firstImgSTD'] = np.std(d['firstImg'])
        # d['firstImg'] = filters.gaussian(d['firstImg'], sigma=d['sigmaImg']) - filters.gaussian(d['firstImg'], sigma=d['sigmaBkgnd'])

        ### parallel
        # compute_parallel(imgMatrix, d, nmbr_processes=3)
        d = findPeaks(d) # Find the peaks in the reference image
        
        if d['recordingMode'] == 'alex':
            # use dummy dict to find spots and integration index on 2nd channel:
            d2['channelNo'] = 2
            d2['integMin'] = d['integMin2']
            d2['integMax'] = d['integMax2']
            d2['seekR'] = d['seekR2']
            d2['integralSpotShape'] = d['integralSpotShape2']
            d2['integR'] = d['integR2']
            d2['AcceptingSpotIntCombo'] = d['AcceptingSpotIntCombo2']
            d2 = findPeaks(d2)
            d2 = intIndexCalc(imgMatrix2, d2)
            # send back data to main dict from dummy dict:
            d['totSpots2'] = d2['totSpots']
            d['xc2'] = d2['xc']
            d['yc2'] = d2['yc']
            d['intIndex2'] = d2['intIndex']
            d['integMin2'] = d2['integMin']
            d['integMax2'] = d2['integMax']
            # use data from both channels to find pairs:
            d = findPairs(d)
            d['xc'] = list(np.asarray(d['xc2']) - d['alexSpotDist'][1])# xc2 and yc2 are the main spot list as we are sure we have all the spots there (D-only list will be added later)
            d['yc'] = list(np.asarray(d['yc2']) - d['alexSpotDist'][0])
            # d['sampleSpot2'] = dC2['sampleSpot']
        d = intIndexCalc(imgMatrix, d)
        if d['recordingMode'] == 'singleChannelDoubleDye':
            d['intIndex2'] = d['intIndex'] #no way to estimate integral index
        d = plot9(d) # FOV with detected spots and integrating area
        d = intensityTracker(imgMatrix, d) # Extract the spots' intensity over different frames
        d = photonStats(d) # do photon statistics
        if d['recordingMode'] == 'alex':
            print('Channel#2 intensity calculation')
            d['channelNo'] = 2
            # use dummy dict to track intensity on 2nd channel:
            d2 = d.copy()
            d2['xc'] = d['xc2']
            d2['yc'] = d['yc2']
            d2['blinkDuration'] = d['blinkDuration2']
            d2['intIndex'] = d['intIndex2']
            # dC2['yc'] = list(np.asarray(d['yc']) + d['alexSpotDist'][0])
            # dC2['xc'] = list(np.asarray(d['xc']) + d['alexSpotDist'][1])
            d2 = intensityTracker(imgMatrix2, d2)
            # send back data to main dict from dummy dict:
            d['intensityTrackAA'] = d2['intensityTrack']
            d['lifeTime2'] = d2['lifeTime']
            d2 = photonStats(d2)
            d['accuPhoton2'] = d2['accuPhoton']
            d['Iavg2'] = d2['Iavg']
            d['nSpots2'] = d2['nSpots']
        ### save data
        from saverAndLoader import saver
        saver(d)
        d['analysisTime'] = time.time()-startTime
        print('Analysis time: ', np.round(d['analysisTime']), 's')
    except Exception as err:
        print(traceback.format_exc())
    # winsound.Beep(frequency=1200, duration=300)
    # winsound.Beep(frequency=1000, duration=300)
    # winsound.Beep(frequency=800, duration=300)

def imgImporter(d):
    from scipy.ndimage import zoom
    ### Reference image reading
    print('Importing image file  >>>  ', d['fnameExt'], end="")
    imgMatrix = tifffile.imread(d['directorySource']+'/'+ d['fnameExt'])
    if (np.size(imgMatrix.shape) < 3): # for single pictures, this prevents from error
        imgMatrix = np.array([imgMatrix,imgMatrix*0])
    imgMatrix = imgMatrix.transpose(1, 2, 0)
    print('  >>>  ' + str(imgMatrix.shape[2])+ ' frames imported')
    # limit frame range:
    if d['LimitFrameRange']:
        imgMatrix = imgMatrix[:,:,d['frameRangeA']:d['frameRangeB']]
    # crop image:
    if d['crop']:
        imgMatrix = imgMatrix[d['cropYA']:d['cropYB'], d['cropXA']:d['cropXB'], :]
    # spatial and temporal binning:
    if d['spBin'] or d['tempBin']:
        imgMatrix = zoom(imgMatrix, (1/d['spBinSize'], 1/d['spBinSize'], 1/d['tempBinSize']))

    d['darkFrameNo'] = 0

    if d['removeFirstDarkFrames']:
        for nFrame in range(d['totFrames']):
            if np.mean(imgMatrix[:, :, nFrame]) < d['darkFrameMeanValueThreshold']:
                d['darkFrameNo'] += 1
            else:
                nFrame = d['totFrames']
        print('First '+ str(d['darkFrameNo']) + ' frames removed')
    imgMatrix = imgMatrix[:, :, d['darkFrameNo']:]
    d['backgroundflag'] = 0
    return (imgMatrix, d)

def findPeaks(d):
    from plotter import plot8, plot9
    import time
    print('Finding spots  >>>  ', end="")
    r = d['seekR']
    img = ndimage.gaussian_filter(d['firstImgs'], sigma=d['sigmaImg']) - ndimage.gaussian_filter(d['firstImgs'], sigma=d['sigmaBkgnd'])
    firstImg = d['firstImg']-ndimage.gaussian_filter(d['firstImg'], sigma=d['sigmaBkgnd'])
    maskImage = 1
    try:
        from numpy import loadtxt
        maskList = loadtxt(d['directorySource']+'/'+ d['fname']+".txt", dtype=str, comments="#", delimiter="\t", unpack=False)
        maskList = np.array([list( map(int,i) ) for i in maskList])
        maskList = maskList[:, [1, 0]] #swap x and y
        maskCircleR = 20 #pixels of mask circle radious
        maskImageIO = np.zeros([img.shape[0],img.shape[1]])
        print('Applying ', np.shape(maskList)[0], ' masks, ', end="")
        maskIndexCircle = [[], []]
        for xx in range(-int(1.1 * maskCircleR), int(1.1 * maskCircleR)):
            for yy in range(-int(1.1 * maskCircleR), int(1.1 * maskCircleR)):
                if (xx ** 2 + yy ** 2) <= maskCircleR ** 2:
                    maskIndexCircle = np.append(maskIndexCircle, [[xx], [yy]], axis=1)
        for listN in range(0, np.shape(maskList)[0]):
            try:
                for ii in range(0, np.shape(maskIndexCircle)[1]):
                    maskImageIO[int(maskList[listN, 0]) + int(maskIndexCircle[1,ii]), int(maskList[listN][1]) + int(maskIndexCircle[0, ii])] = 1
            except:
                pass#print(traceback.format_exc())
        img = np.multiply(img, maskImageIO)
        d['maskIndexCircle']=maskIndexCircle
        d['maskList']=maskList
        d['maskCircleR']=maskCircleR
    except:
        pass
    intR = d['integR']
    acceptR = 2 * intR
    if d['recordingMode'] == 'alex':
        acceptR = 2 * d['integR'] + np.sqrt(d['Ch2relativeDeviation'][0]**2+d['Ch2relativeDeviation'][1]**2) + 1
    thresholdMin = 2 * np.std(img)
    thresholdMax = 3000 * np.std(img)
    print('I = [',int(thresholdMin), ',', int(thresholdMax), '],', end="")
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    mask = x ** 2 + y ** 2 <= r ** 2
    data_max = scfilters.maximum_filter(img, footprint=mask)
    maxima = (img == data_max)
    # data_min = scfilters.minimum_filter(img, footprint=mask)
    # diff = ((data_max - data_min) > thresholdMin)
    diff = (data_max > thresholdMin)
    maxima[diff == 0] = 0
    # diff = (data_max < thresholdMax) # to avoid beads that are much bright
    maxima[diff == 0] = 0
    labeled, totSpots = ndimage.label(maxima)
    # numObjTot = []
    # numObjTot.append(totSpots)  # to save the number of spots in different frames
    slices = ndimage.find_objects(labeled)
    xc, yc = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        if (acceptR < x_center < img.shape[1]-acceptR) & (acceptR < y_center < img.shape[0]-acceptR):
            xc.append(x_center)
            yc.append(y_center)
    totSpots = len(xc)
    

    
    # filter spots based on integrated intensity
    intIndex = np.zeros([2, 1])
    for xx in range(-int(1.3 * intR), int(1.3 * intR)):
        for yy in range(-int(1.3 * intR), int(1.3 * intR)):
            if xx ** 2 + yy ** 2 <= intR ** 2:
                intIndex = np.append(intIndex, [[xx], [yy]], axis=1)
    intIndex=intIndex[:, 1:-1]
    
    # select spots interactively based on integrated intensity
    intensity = np.zeros([totSpots, 1])
    for sN in range(-totSpots + 1, 0):
        beadN = int(-sN)
        for pxl in range(intIndex.shape[1]):
            intensity[beadN] += img[int(yc[beadN]+intIndex[1, pxl]), int(xc[beadN]+intIndex[0, pxl])]
    if d['AcceptingSpotIntCombo'] == 0:
        d = plot8select(d, intensity)
    intensityOutlier = np.zeros([totSpots, 1])
    intensityOutlier[:] = np.NaN
    for sN in range(-totSpots + 1, 0):
        beadN = int(-sN)
        if (intensity[beadN] > d['integMax']) or (intensity[beadN] < d['integMin']):
            xc.pop(beadN)
            yc.pop(beadN)
            intensityOutlier[beadN] = intensity[beadN]
    totSpots = len(xc)
    d = plot8(d, intensity, intensityOutlier) # to plot the histogram to compare outlier spots to good spots
    d['SNR'] = np.zeros(totSpots)
    d['SNRreal'] = np.zeros(totSpots)
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    # STDmask =  (x ** 2 + y ** 2 <= r ** 2) & (x ** 2 + y ** 2 > intR ** 2)
    d['firstImgsSTD'] = np.std(img)
    for s in range(totSpots):
        # subimg = img[int(xc[s]-r):int(xc[s]+r)+1, int(yc[s]-r):int(yc[s]+r)+1]
        d['SNR'][s] = img[int(yc[s]), int(xc[s])]/d['firstImgsSTD']
        d['SNRreal'][s] = firstImg[int(yc[s]), int(xc[s])]/d['firstImgSTD']
    print('  >>>  ', totSpots, ' spots detected, ', end="")
    print('Detection SNR =', round(np.mean(d['SNR']), 1), "±", round(np.std(d['SNR']), 1), end="")
    print('(Real SNR =', round(np.mean(d['SNRreal']), 1), "±", round(np.std(d['SNRreal']), 1),")")
    d['totSpots'] = totSpots
    d['xc'] = xc
    d['yc'] = yc
    return (d)

def trackPosition(img, r, xc , yc, lifeTime, successTot):
    totSpots = np.shape(x, 0)
    pointCloud = np.append([x], [y], axis=0)
    pointCloud = np.transpose(pointCloud)
    success = 0
    for beadN in range(0, totSpots):
        pt = [x[beadN], y[beadN]]
        distance, index = spatial.KDTree(pointCloud).query(pt, k=3)
        if distance[0] < r:
            xc = pointCloud[index, 0]
            yc = pointCloud[index, 1]
            lifeTime[beadN] += 1
            success = 1
        print('Successive tracks:', success)
        successTot.append(success)
    return (xc , yc, lifeTime, successTot)

def findPairs(d):# called in ALEX mode
    maxMatchingDist = 5 # maximum distance between the expected point location and the point in the point cloud
    totSpots = d['totSpots']
    totSpots2 = d['totSpots2']
    d['spotLabel'] = np.zeros(d['totSpots2'])
    x = d['xc']
    y = d['yc']
    x2 = d['xc2']
    y2  = d['yc2']
    pointcloud = np.append([x], [y], axis=0)
    pointcloud = np.transpose(pointcloud)
    pointcloud2 = np.append([x2], [y2], axis=0)
    pointcloud2 = np.transpose(pointcloud2)
    for sn in range(0, totSpots2): # all the possible spots will shine in Ch.2
        ptD=[x2[sn]-d['Ch2relativeDeviation'][1], y2[sn]-d['Ch2relativeDeviation'][0]] # expected point for donor in Ch.1
        distanceD, indexD = spatial.KDTree(pointcloud).query(ptD, k=3)
        ptA=[x2[sn], y2[sn]] # expected point for acceptor in Ch.1
        distanceA, indexA = spatial.KDTree(pointcloud).query(ptA, k=3)
        if (distanceD[0] <= maxMatchingDist) or (distanceA[0] <= maxMatchingDist):
            # spotDist[sn,:] = [x2[sn] - x[indexD[0]], y2[sn] - y[indexD[0]]]
            d['spotLabel'][sn] += 3
        else:
            d['spotLabel'][sn] += 2
    for sn in range(0, totSpots):#to find D-only spots
        pt2 = [x[sn] + d['Ch2relativeDeviation'][1], y[sn] + d['Ch2relativeDeviation'][0]]#expected position for acceptor in Ch.2
        distance, index = spatial.KDTree(pointcloud2).query(pt2, k=3)
        if distance[0] > maxMatchingDist:
            pt2 = [x[sn], y[sn]]#maybe this is a hi-FRET, expected position for acceptor in Ch.2
            distance, index = spatial.KDTree(pointcloud2).query(pt2, k=3)
            if distance[0] > maxMatchingDist:
                d['spotLabel'] = np.append(d['spotLabel'], 1)
                d['xc2'].append(x[sn] + d['Ch2relativeDeviation'][1])
                d['yc2'].append(y[sn] + d['Ch2relativeDeviation'][0])
    d['totSpots'] = len(d['xc2'])
    return(d)

# @njit
def intensityTracker(imgMatrix, d):
    from GUIfunc import status
    d['backgroundflag'] = 0
    yc = d['yc']
    xc = d['xc']
    totSpots = d['totSpots']
    totFrames = imgMatrix.shape[2]
    index2photon = d['index2photon']
    blinkDuration = d['blinkDuration']
    intIndex = d['intIndex']
    intensityTrack = np.zeros((totSpots, totFrames)).astype(np.float32)
    intensityTrackDA = np.zeros((totSpots, totFrames)).astype(np.float32)
    lifeTime = np.ones(totSpots)
    opportunity = np.ones(totSpots) * blinkDuration
    for nFrame in range(totFrames):
        print('Frame #', nFrame + 1)
        status('Frame #'+ str(nFrame + 1))
        img = index2photon * imgMatrix[:, :, nFrame]
        d = background(img, d)
        img = img - d['bkgnd']
        if (d['corrIlluPattern']) & (d['channelNo'] == 2): # to compensate variations in Ch.2 illumination pattern in comparison to Ch.1
            img = img * d['bkgnd1']/d['bkgnd2']
        for beadN in range(totSpots):# Intensity integration
            #mask = (xArr[np.newaxis, :] - cx) ** 2 + (yArr[:, np.newaxis] - cy) ** 2 < intR ** 2
            # intensityTrack[beadN, i] = np.sum(img[mask[:,:,beadN]])
            for pxl in range(intIndex.shape[1]):# integration for DexDem or AexAem
                intensityTrack[beadN, nFrame] += img[int(yc[beadN]+intIndex[1, pxl]), int(xc[beadN]+intIndex[0, pxl])]
            if d['channelNo'] == 1: # integration for DexAem
                for pxl in range(d['intIndex2'].shape[1]):
                    intensityTrackDA[beadN, nFrame] += img[int(yc[beadN]+d['intIndex2'][1, pxl]+d['Ch2relativeDeviation'][0]), int(xc[beadN]+d['intIndex2'][0, pxl]+d['Ch2relativeDeviation'][1])]
            if (intensityTrack[beadN, nFrame] > d['integMin']) & (opportunity[beadN] > 0):#if (intensityTrack[beadN, nFrame] > 0.5 * (np.mean(intensityTrack[beadN, 0:int(lifeTime[beadN])+1]))) & (opportunity[beadN] > 0):
                lifeTime[beadN] += blinkDuration - opportunity[beadN]  + 1
                opportunity[beadN] = blinkDuration
            else:
                opportunity[beadN] = opportunity[beadN] - 1
    d['lifeTime'] = lifeTime
    d['intensityTrack'] = intensityTrack
    if d['channelNo'] == 1:
        d['intensityTrackDA'] = intensityTrackDA
    return(d)
    ### make a gif image of changes in spots
    # figNo = imgGifShow(img, figNo, x, y)



def imgGifShow(img, figNo, x, y):
    totSpots = np.shape(x)
    #imgEq = exposure.equalize_hist(img/np.max(img))
    #imgEq = exposure.equalize_adapthist(img/np.max(img), clip_limit=0.1)
    imgEq = 2 * img/np.max(img)
    # plt.figure(figNo)
    # figNo += 1
    # camera = Camera(fig)
    # plt.clf()
    plt.imshow(imgEq, cmap='gray')
    # plt.scatter(x, y, s=25, c='none', marker='o', edgecolors='red' , linewidths=1)
    for c in range(totSpots):
        if alive[c] == 1:
            tColor = 'red'
        else:
            tColor = 'w'
        plt.scatter(x[c], y[c], s=25, c='none', marker='o', edgecolors=tColor, linewidths=1)
        # plt.text(x[c], y[c], str(lifeTime[c]), c=tColor,
        #         horizontalalignment='center', verticalalignment='bottom')
    # plt.show()
    # camera.snap()
    # #plt.draw()
    # plt.pause(0.01)
    # plt.show()
    return(figNo)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def photonStats(d):
    d['accuPhoton'] = np.empty(d['totSpots'])
    for ns in range(0, d['totSpots']):
        d['accuPhoton'][ns] = np.sum(d['intensityTrack'][ns,0:int(d['lifeTime'][ns])],0)
    Iavg = np.zeros(d['totFrames'])
    for nf in range(d['totFrames']):
        nn = 0
        for ns in range(d['totSpots']):
            if d['lifeTime'][ns] > nf:
                Iavg[nf] += d['intensityTrack'][ns, nf]
                nn += 1
        if nn > 0:
            Iavg[nf] = Iavg[nf]/nn
    # Iavg = Iavg[Iavg != 0]
    d['Iavg'] = Iavg #average of photons per frame
    # plot number of alive dyes vs time (Ibrahim)
    d['nSpots'] = np.zeros(d['totFrames'])
    for nf in range(d['totFrames']):
        for ns in range(d['totSpots']):
            if d['lifeTime'][ns] >= nf:
                d['nSpots'][nf] += 1
    return (d)

def background(img, d):
    imgForBkGnd = img.copy()
    if d['realBkgnd']:
        print ('calculating background by removing spots!')
        intIndex = d['intIndexAvg']
        totSpots = d['totSpots']
        yc = d['yc']
        xc = d['xc']
        BkGndIndex = d['marginIndex']
        sign = 1
        if d['channelNo'] == 2 : sign = -1
        for sn in range(totSpots):
            spotBkGnd = 0
            c = 0
            for pxl in range(BkGndIndex.shape[1]):
                if (0 < (yc[sn]+BkGndIndex[1, pxl]) < d['imgY']) & (0 < (xc[sn]+BkGndIndex[0, pxl]) < d['imgX']): # check if the index is within image borders
                    spotBkGnd += img [int(yc[sn]+BkGndIndex[1, pxl]), int(xc[sn]+BkGndIndex[0, pxl])]
                    c += 1
            spotBkGnd = spotBkGnd/c
            for pxl in range(intIndex.shape[1]):
                imgForBkGnd[int(yc[sn]+intIndex[1, pxl]), int(xc[sn]+intIndex[0, pxl])] = spotBkGnd
                imgForBkGnd[int(yc[sn]+intIndex[1, pxl]+ sign * d['Ch2relativeDeviation'][0]), int(xc[sn] + intIndex[0, pxl]+ sign * d['Ch2relativeDeviation'][1])] = spotBkGnd
    d['bkgnd'] = ndimage.gaussian_filter(imgForBkGnd, sigma=d['sigmaBkgnd'])
    #d['bkgnd'] = filters.gaussian(img, sigma=d['sigma2'])
    if d['backgroundflag'] == 0: #save background
        if d['channelNo'] == 1:
            d['bkgnd1']=d['bkgnd']
        if d['channelNo'] == 2:
            d['bkgnd2']=d['bkgnd']
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        plt.imshow(d['bkgnd'], cmap='gray')
        plt.title('Background, Average= ' + str(int(np.mean(d['bkgnd']))) + '±' + str(int(np.std(d['bkgnd'])))+ ' photons')
        if d['savePlots']:
            plt.savefig(d['directorySave'] + d['fname'] + ' Background-ch' + str(d['channelNo']) + d['savePlotsFormat'])
        # plt.show()
        plt.close(fig)
        d['backgroundflag'] = 1 #to save the background figure just one time
    return (d)
