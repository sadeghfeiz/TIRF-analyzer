def intIndexCalc(imgMatrix, d):
    import matplotlib.pyplot as plt
    import numpy as np
    import lmfit
    from lmfit.models import gaussian
    from scipy.special import erf
    from scipy.interpolate import griddata
    import scipy.ndimage as ndimage
    from matplotlib.collections import PatchCollection
    # Circle index
    intIndexCirc = [[],[]]
    intR = d['integR']
    for xx in range(-int(1.3 * intR), int(1.3 * intR)):
        for yy in range(-int(1.3 * intR), int(1.3 * intR)):
            if xx ** 2 + yy ** 2 <= intR ** 2:
                intIndexCirc = np.append(intIndexCirc, [[xx], [yy]], axis=1)
    # index by average value of spots
    intIndexAvg = [[],[]]
    dy = 7
    dx = 7
    threshAvg = 0.2
    threshAvg2 = threshAvg/2 # define a ring around the spot for background calculations
    sampleImg =np.zeros([2*dy, 2*dx])
    img = np.int32(imgMatrix[:, :, 0])
    img = img-ndimage.gaussian_filter(img, sigma=d['sigmaBkgnd'])
    for sn in range(d['totSpots']):
        sampleImg += (img[int(d['yc'][sn]-dy):int(d['yc'][sn]+dy),int(d['xc'][sn]-dx):int(d['xc'][sn]+dx)]) / d['totSpots']
    backgnd = np.mean([sampleImg[0,-1],sampleImg[-1,-1],sampleImg[0,0],sampleImg[-1,0]])
    for xx in range(-int(dx), int(dx)):
        for yy in range(-int(dy), int(dy)):
            if (sampleImg[yy+dy,xx+dx]-backgnd) > (threshAvg * (sampleImg[dy,dx]-backgnd)):
                intIndexAvg = np.append(intIndexAvg, [[xx], [yy]], axis=1)
    d['intIndexAvg'] = intIndexAvg #reserved for removing spot for backgroung calculations
    d['marginIndex'] = marginPixels(intIndexAvg) #reserved for removing spot for backgroung calculations

    # index by fitting skewed Gaussian fit on the spots
    threshFit = 0.1
    def SkewedGaussian2D(x, y, amplitude, centerx, centery, sigmax, sigmay, gammax, gammay, offset):
        tiny = 1.0e-15
        s2 = np.sqrt(2.0)
        asymx = 1 + erf(gammax*(x-centerx)/max(tiny, (s2*sigmax)))
        asymy = 1 + erf(gammay*(y-centery)/max(tiny, (s2*sigmay)))
        return offset + asymx * asymy * gaussian(x, amplitude, centerx, sigmax) * gaussian(y, amplitude, centery, sigmay)
    x, y = np.meshgrid(range(2*dx),range(2*dy))
    model = lmfit.Model(SkewedGaussian2D, independent_vars=['x', 'y'])
    params = model.make_params(amplitude=np.max(sampleImg), offset =0)
    params['centery'].set(value=dy, min=dy/2, max=3*dy/2)
    params['centerx'].set(value=dx, min=dx/2, max=3*dx/2)
    params['sigmax'].set(value=1, min=0)
    params['sigmay'].set(value=1, min=0)
    params['gammax'].set(value=0, min=-10, max=10)
    params['gammay'].set(value=0, min=-10, max=10)
    result = model.fit(sampleImg, x=x, y=y, params=params)
    #lmfit.report_fit(result)
    fit = model.func(x, y, **result.best_values)
    intIndexFit = [[],[]]
    for xx in range(-int(dx), int(dx)):
        for yy in range(-int(dy), int(dy)):
            if fit[yy+dy,xx+dx]-result.best_values['offset'] > threshFit * (fit[int(result.best_values['centery']),int(result.best_values['centerx'])]-result.best_values['offset']):
                intIndexFit = np.append(intIndexFit, [[xx], [yy]], axis=1)

    if d['integralSpotShape'] == 'circle':
        d['intIndex'] = intIndexCirc
    if d['integralSpotShape'] == 'average':
        d['intIndex'] = intIndexAvg
    if d['integralSpotShape'] == 'fit':
        d['intIndex'] = intIndexFit
    # plotting
    fig, axs = plt.subplots(1, 3)#, figsize=(7, 28))
    plt.rcParams["axes.titlesize"] = 8
    vmax = np.max(sampleImg)
    ax = axs[0]
    ax.imshow(sampleImg, vmin=0, vmax=vmax)
    ax.set_title('Circle on average')
    patches = pathAround(intIndexCirc,[dx],[dy])
    collection = PatchCollection(patches, edgecolor= 'orange', linewidths=1)
    ax.add_collection(collection)
    ax = axs[1]
    ax.imshow(sampleImg, vmin=0, vmax=vmax)
    ax.set_title('Threshold on average')
    patches = pathAround(intIndexAvg,[dx],[dy])
    collection = PatchCollection(patches, edgecolor= 'orange', linewidths=1)
    ax.add_collection(collection)
    # ax.scatter(d['marginIndex'][0]+dx,d['marginIndex'][1]+dy,marker='.',c='chocolate', s=1)
    ax = axs[2]
    ax.imshow(fit, vmin=0, vmax=vmax)
    try:
        patches = pathAround(intIndexFit,[dx],[dy])
        collection = PatchCollection(patches, edgecolor= 'orange', linewidths=1)
        ax.add_collection(collection)
    except:
        pass
    ax.set_title('Threshold on fit')


    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    # plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname'] + ' sample spot-ch' + str(d['channelNo']) + d['savePlotsFormat'], bbox_inches='tight')
    plt.close()

    return(d)
def pathAround(intIndex,xcList,ycList):
    import numpy as np
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    pairs = zip(np.asarray(xcList), np.asarray(ycList))
    Path = mpath.Path
    patches =[]
    try:
        for xc, yc in pairs:
            for ii in range(intIndex.shape[1]):
                for side in [-1,1]:
                    if not(intIndex[0, ii] + side in intIndex[0, intIndex[1, :] == intIndex[1, ii]]):
                        path_data=[(Path.MOVETO,[xc + intIndex[0, ii] + side * 0.5, yc + intIndex[1, ii] - 0.5])]
                        path_data.append((Path.LINETO,[xc + intIndex[0, ii] + side * 0.5, yc + intIndex[1, ii] + 0.5]))
                        codes, verts = zip(*path_data)
                        patch = mpatches.PathPatch(mpath.Path(verts, codes))
                        patches.append(patch)
                    if not(intIndex[1, ii] + side in intIndex[1, intIndex[0, :] == intIndex[0, ii]]):
                        path_data=[(Path.MOVETO,[xc + intIndex[0, ii] - 0.5, yc + intIndex[1, ii] +side * 0.5])]
                        path_data.append((Path.LINETO,[xc + intIndex[0, ii] + 0.5, yc + intIndex[1, ii] + side * 0.5]))
                        codes, verts = zip(*path_data)
                        patch = mpatches.PathPatch(mpath.Path(verts, codes))
                        patches.append(patch)
    except:
        pass
    return patches

def marginPixels(intIndex):#pixels around the spot for background calculations
    import numpy as np
    marginIndex = [[],[]]
    try:
        for ii in range(intIndex.shape[1]):
            for side in [-1,1]:
                if not(intIndex[0, ii] + side in intIndex[0, intIndex[1, :] == intIndex[1, ii]]):
                    marginIndex = np.append(marginIndex, [[intIndex[0, ii] + 2 * side], [intIndex[1, ii]]], axis=1)
    except:
        pass
    return marginIndex