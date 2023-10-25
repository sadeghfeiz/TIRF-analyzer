from imgProc import *
from intIndexCalc import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imgProc import *
import scipy.optimize as opt


def plot1(d):
    ### life-time histogram
    # histRange = [0, 250]
    lifeTime = d['lifeTime']
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    # n, bins, patches = plt.hist(lifeTime, bins=30, range=histRange , align='mid', facecolor='g', alpha=0.75, )
    n, bins, patches = plt.hist(lifeTime, bins=30, align='mid', facecolor='grey', alpha=0.75, )
    plt.xlabel('lifetime (frame)')
    plt.ylabel('Count')
    plt.title('Average: ' + str(int(np.mean(lifeTime))) + '±' + str(int(np.std(lifeTime)))+ ' frames')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' lifetime hist' + d['savePlotsFormat'], bbox_inches='tight')
    plt.close(fig)
    
    try:
        lifeTime = d['lifeTime2']
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        # n, bins, patches = plt.hist(lifeTime, bins=30, range=histRange , align='mid', facecolor='g', alpha=0.75, )
        n, bins, patches = plt.hist(lifeTime, bins=30, align='mid', facecolor='grey', alpha=0.75, )
        plt.xlabel('lifetime (frame)')
        plt.ylabel('Count')
        plt.title('Average: ' + str(int(np.mean(lifeTime))) + '±' + str(int(np.std(lifeTime)))+ ' frames')
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave']+d['fname']+ ' lifetime hist-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        plt.close(fig)
    except:
        pass
    
    return(d)

def plot2(d):
    ### photon count histogram
    # histRange = [0, 2000]
    import matplotlib.ticker as ticker
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    # n, bins, patches = plt.hist(d['accuPhoton'],range= histRange, bins=int(50) + 1, align='mid', facecolor='grey', alpha=0.5, )
    n, bins, patches = plt.hist(d['accuPhoton'], bins=int(30) + 1, align='mid', facecolor='grey', alpha=0.5, )
    # plt.xlim([0,20000])
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.xlabel('Total received photons per spot')
    plt.ylabel('Count')
    # plt.title('Average: ' + str(int(np.mean(d['accuPhoton']))) +'±' + str(int(np.std(d['accuPhoton'])))+ ' photons received in total per spot')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' Received photon per spot' + d['savePlotsFormat'], bbox_inches='tight')

    plt.close(fig)
    try:
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        # n, bins, patches = plt.hist(d['accuPhoton'],range= histRange, bins=int(50) + 1, align='mid', facecolor='grey', alpha=0.5, )
        n, bins, patches = plt.hist(d['accuPhoton2'], bins=int(30) + 1, align='mid', facecolor='grey', alpha=0.5, )
        # plt.xlim([0,20000])
        plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
        plt.xlabel('Total received photons per spot')
        plt.ylabel('Count')
        # plt.title('Average: ' + str(int(np.mean(d['accuPhoton']))) +'±' + str(int(np.std(d['accuPhoton'])))+ ' photons received in total per spot')
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave']+d['fname']+ ' Received photon per spot-ch2' + d['savePlotsFormat'], bbox_inches='tight')

        plt.close(fig)
    except:
        pass
    return(d)

def plot3(d):
    ### Average No. of photons / frame / spot (David suggestion)
    # histRange = [150, 300]
    nBins = 30
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    # n, bins, patches = plt.hist(d['Iavg'], bins=int(100), range=histRange, align='mid', facecolor='grey', alpha=0.5)
    n, bins, patches = plt.hist(d['Iavg'], bins=nBins, align='mid', facecolor='grey', alpha=0.5)
    plt.xlabel('Average spot intensity per frame (photons)')
    plt.ylabel('Count')
    plt.title('Average: ' + str(int(np.mean(d['Iavg']))) + '±' + str(int(np.std(d['Iavg']))) + ' photons per frame per spot')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' Avg photons per frame per spot' + d['savePlotsFormat'], bbox_inches='tight')
    plt.close(fig)
    try:
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        # n, bins, patches = plt.hist(d['Iavg'], bins=int(100), range=histRange, align='mid', facecolor='grey', alpha=0.5)
        n, bins, patches = plt.hist(d['Iavg2'], bins=nBins, align='mid', facecolor='grey', alpha=0.5)
        plt.xlabel('Average spot intensity per frame (photons)')
        plt.ylabel('Count')
        plt.title('Average: ' + str(int(np.mean(d['Iavg2']))) + '±' + str(int(np.std(d['Iavg2']))) + ' photons per frame per spot')
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave']+d['fname']+ ' Avg photons per frame per spot-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        plt.close(fig)
    except:
        pass
    return(d)

def plot4(d):
    ### spots intensity histogram (David suggestion)
    ### fitting a surface to the spots mean intensity to see homeogenity of FOV (David suggestion)
    xc = d['xc']
    yc = d['yc']
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    
    IAvgSpot = np.empty(d['totSpots'])
    for ns in range(d['totSpots']):
        IAvgSpot[ns] = np.mean(d['intensityTrack'][ns, 0:int(d['lifeTime'][ns])])
    # ??????? n, bins, patches = plt.hist(d['Iavg'], bins=int(30), facecolor='g', alpha=0.75, )
    n, bins, patches = plt.hist(IAvgSpot, bins=int(30), facecolor='grey', alpha=0.75, )
    plt.xlabel('Spots intensity (photon)')
    plt.ylabel('Count')
    plt.title('Average: ' + str(int(np.mean(IAvgSpot))) + '±' + str(int(np.std(IAvgSpot))) + ' photons per spot per frame')#str(int(np.mean(Iavg)))
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' intensity of spots' + d['savePlotsFormat'], bbox_inches='tight')
    plt.close(fig)
     ### convert data into proper format
    x_data = xc
    y_data = yc
    z_data = IAvgSpot
    try:
        xc = d['xc2']
        yc = d['yc2']
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        
        IAvgSpot = np.empty(d['totSpots2'])
        for ns in range(d['totSpots2']):
            IAvgSpot[ns] = np.mean(d['intensityTrackAA'][ns, 0:int(d['lifeTime2'][ns])])
        # ??????? n, bins, patches = plt.hist(d['Iavg'], bins=int(30), facecolor='g', alpha=0.75, )
        n, bins, patches = plt.hist(IAvgSpot, bins=int(30), facecolor='grey', alpha=0.75, )
        plt.xlabel('Spots intensity (photon)')
        plt.ylabel('Count')
        plt.title('Average: ' + str(int(np.mean(IAvgSpot))) + '±' + str(int(np.std(IAvgSpot))) + ' photons per spot per frame')#str(int(np.mean(Iavg)))
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave']+d['fname']+ ' intensity of spots-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        plt.close(fig)
    except:
        pass
    return(d)

def plot5(d): # intensity track plot
    maxIntensity = 2 * np.mean(np.mean(d['intensityTrack'][:, 0:10]))
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    d['plot5CMap'] = ''  # 'gray' , otherwise it automatically choose 'viridis'
    d = customCMap(d)
    inesities = np.mean(d['intensityTrack'][:, :], 1)
    sortIndex = np.argsort(-inesities)
    if d['frameRangePlot']:
        d['intensityTrack'] = d['intensityTrack'][:,d['intensityTrackA']:d['intensityTrackB']]
    plt.imshow(d['intensityTrack'][sortIndex,:], origin='lower', aspect='auto', cmap=d['newCMap'], vmin = 0, vmax = 1.5 * d['integMax']) #cmap=d['newCMap'] vmax=5 * np.mean(d['intensityTrack']))
    cbar = plt.colorbar(pad=0.02)
    cbar.set_label('# of photons/spot/frame', rotation=270, labelpad=15)
    if d['markEvent']:
        plt.plot((d['markEventFrame'], d['markEventFrame']), (0, d['totSpots']-1),  c= 'black', linestyle='--', linewidth=.5)
    # plt.scatter(d['lifeTime'][sortIndex], range(0, d['totSpots']), s= 1, c= 'red', marker='.')
    plt.xlim([0,d['totFrames']])
    plt.xlabel('Frame #')
    plt.ylabel('Spot # (sorted by intensity)')
    # plt.title('Detection SNR= ' + str(round(np.mean(d['SNR']),2)) + '±' + str(round(np.std(d['SNR']),2))+ ', real SNR= '+ str(round(np.mean(d['SNRreal']),2)) + '±' + str(round(np.std(d['SNRreal']),2)))
    # fig.set_size_inches(5, 5)
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave'] + d['fname'] + ' intensity track_intSort' + d['savePlotsFormat'], bbox_inches='tight')
    # sort by intensity
    fig2 = plt.figure(d['figNo'])
    d['figNo'] += 1
    sortIndex = np.argsort(-d['lifeTime'])
    plt.imshow(d['intensityTrack'][sortIndex,:], origin='lower', aspect='auto',cmap=d['newCMap'] ,vmin = 0, vmax = 1.5 * d['integMax']) #vmax=maxIntensity # vmax=5 * np.mean(d['intensityTrack']))
    #plt.text(lifeTime, totSpots,  '%d' % int(accuPhoton))
    cbar = plt.colorbar(pad=0.02)
    cbar.set_label('# of photons/spot/frame', rotation=270, labelpad=15)
    if d['markEvent']:
        plt.scatter(d['markEventFrame'], 0, marker='^',  c= 'red')
    plt.scatter(d['lifeTime'][sortIndex], range(0, d['totSpots']), s= 1, c= 'red', marker='.')
    plt.xlim([0,d['totFrames']])
    plt.xlabel('Frame #')
    plt.ylabel('Spot # (sorted by life time)')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' intensity track_lifeSort' + d['savePlotsFormat'], bbox_inches='tight')

    # plt.show()
    plt.close(fig) 
    plt.close(fig2)
    try:
        trymaxIntensity = 2 * np.mean(np.mean(d['intensityTrackAA'][:, 0:10]))
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        d['plot5CMap'] = ''   #'gray' , otherwise it automatically choose 'viridis'
        d = customCMap(d)
        inesities = np.mean(d['intensityTrackAA'][:, :], 1)
        sortIndex = np.argsort(-inesities)
        if d['frameRangePlot']:
            d['intensityTrackAA'] = d['intensityTrackAA'][:,d['intensityTrackA2']:d['intensityTrackB2']]
        plt.imshow(d['intensityTrackAA'][sortIndex,:], origin='lower', aspect='auto', cmap=d['newCMap'], vmin = 0, vmax = 1.5 * d['integMax2']) #cmap=d['newCMap'] vmax=5 * np.mean(d['intensityTrack']))
        cbar = plt.colorbar(pad=0.02)
        cbar.set_label('# of photons/spot/frame', rotation=270, labelpad=15)
        if d['markEvent']:
            plt.plot((d['markEventFrame'], d['markEventFrame']), (0, d['totSpots2']-1),  c= 'black', linestyle='--', linewidth=.5)
        # plt.scatter(d['lifeTime'][sortIndex], range(0, d['totSpots']), s= 1, c= 'red', marker='.')
        plt.xlim([0,d['totFrames']])
        plt.xlabel('Frame #')
        plt.ylabel('Spot # (sorted by intensity)')
        # plt.title('Detection SNR= ' + str(round(np.mean(d['SNR']),2)) + '±' + str(round(np.std(d['SNR']),2))+ ', real SNR= '+ str(round(np.mean(d['SNRreal']),2)) + '±' + str(round(np.std(d['SNRreal']),2)))
        # fig.set_size_inches(5, 5)
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave'] + d['fname'] + ' intensity track_intSort-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        # sort by intensity
        fig2 = plt.figure(d['figNo'])
        d['figNo'] += 1
        sortIndex = np.argsort(-d['lifeTime2'])
        plt.imshow(d['intensityTrackAA'][sortIndex,:], origin='lower', aspect='auto',cmap=d['newCMap'] ,vmin = 0, vmax = 1.5 * d['integMax2']) #vmax=maxIntensity # vmax=5 * np.mean(d['intensityTrack']))
        #plt.text(lifeTime, totSpots,  '%d' % int(accuPhoton))
        cbar = plt.colorbar(pad=0.02)
        cbar.set_label('# of photons/spot/frame', rotation=270, labelpad=15)
        if d['markEvent']:
            plt.scatter(d['markEventFrame'], 0, marker='^',  c= 'red')
        # plt.scatter(d['lifeTime2'][sortIndex], range(0, d['totSpots2']), s= 1, c= 'red', marker='.')
        plt.xlim([0,d['totFrames2']])
        plt.xlabel('Frame #')
        plt.ylabel('Spot # (sorted by life time)')
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave'] + d['fname']+ ' intensity track_lifeSort-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        plt.close(fig2)
    except Exception as e: 
        print(e)
        
    return(d)

def plot6(d):
    ### spots vs time (Ibrahim)
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    if d['frameRangePlot']:
        d['nSpots'] = d['nSpots'][d['frameRangePlotA']:d['frameRangePlotB']]
    plt.plot(d['nSpots'])
    plt.xlabel('Frame')
    plt.ylabel('# of spots')

    # Fit function
    def fitExponential(x, amplitude, tau, offset):
        y = amplitude * np.exp(-x/tau) 
        return y.ravel()
    xFit=np.arange(d['FFS'], d['nSpots'].shape[0]-d['blinkDuration'])
    iniGuess = (d['nSpots'][d['FFS']], np.mean(d['lifeTime']), 0)
    parameters, covariance = opt.curve_fit(fitExponential, xdata=xFit, ydata=d['nSpots'][d['FFS']:-d['blinkDuration']], p0=iniGuess)
    plt.plot(xFit, fitExponential(xFit, *parameters))
    plt.title('Characteristic time = ' + str(int(parameters[1])) + ' frames')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname']+ ' survived spots' + d['savePlotsFormat'], bbox_inches='tight')
    plt.close(fig)
    try:
        fig = plt.figure(d['figNo'])
        d['figNo'] += 1
        if d['frameRangePlot']:
            d['nSpots2'] = d['nSpots2'][d['frameRangePlotA']:d['frameRangePlotB']]
        plt.plot(d['nSpots2'])
        plt.xlabel('Frame')
        plt.ylabel('# of spots')

        # Fit function
        def fitExponential(x, amplitude, tau, offset):
            y = amplitude * np.exp(-x/tau) 
            return y.ravel()
        xFit=np.arange(d['FFS'], d['nSpots2'].shape[0]-d['blinkDuration2'])
        iniGuess = (d['nSpots2'][d['FFS']], np.mean(d['lifeTime2']), 0)
        parameters, covariance = opt.curve_fit(fitExponential, xdata=xFit, ydata=d['nSpots2'][d['FFS']:-d['blinkDuration2']], p0=iniGuess)
        plt.plot(xFit, fitExponential(xFit, *parameters))
        plt.title('Characteristic time = ' + str(int(parameters[1])) + ' frames')
        plt.gca().set_box_aspect(1)
        if d['showPlots']:
            plt.show()
        if d['savePlots']:
            plt.savefig(d['directorySave']+d['fname']+ ' survived spots-ch2' + d['savePlotsFormat'], bbox_inches='tight')
        plt.close(fig)
    except:
        pass
    return(d)

def plot7(d):
    ### intensity track one by one
    from imgProc import moving_average
    from pathlib import Path
    print('Plotting intensity track one by one...')
    showFold = d['showFolds']
    if d['frameRangePlot']:
        d['intensityTrack'] = d['intensityTrack'][d['frameRangePlotA']:d['frameRangePlotB']]
    # maxY = intensityTrack.max()
    maxY = np.round(1.3 * np.maximum(d['integMax'],d['integMax2']), decimals=-2)
    #fig, axs = plt.subplots(2,1)
    fig, axs = plt.subplots(1, 1)
    # fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    #ax=axs[0]
    ax = axs
    if showFold:
        ax2 = ax.twinx()
        ax2.set_ylabel('Fold', color='red')
        ax2.grid(color='r', linestyle='--', linewidth=0.3)
    ax3 = ax.twiny()
    ax3.set_xlabel('frames')
    for sn in range(d['totSpots']):
        if 1:#(d['lifeTime'][sn] > d['intensityTrack'].shape[1] / 3):
            fold1 = np.mean(d['intensityTrack'][sn, 0:10])
            ax.set_title('Spot #' + str(sn))
            # ax.set_xlabel('Frame #')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('# of photons')# ax = plt.gca()
            ax.set_ylim([0, maxY])
            # ax.set_xlim([0, d['intensityTrack'].shape[1]])
            ax.set_xlim([0, d['intensityTrack'].shape[1]/d['FPS']])
            if showFold:
                ax2.set_yticks(np.arange(0, maxY/fold1+1, 1.0))
                ax2.set_ylim([0, maxY/fold1])
                ax2.tick_params(axis='y', colors='red')
            ax3.set_xlim([0, d['intensityTrack'].shape[1]])
            if d['markEvent']:
                d['markEventFrame2'] = 105
                ax3.plot((d['markEventFrame'], d['markEventFrame']),(0, maxY),c='lightgrey', ls=':')
                ax3.plot((d['markEventFrame2'], d['markEventFrame2']), (0, maxY), c='lightgrey', ls=':')
            xRange = np.arange(0,d['intensityTrack'].shape[1]/d['FPS'],1/d['FPS'])
            ax.plot(xRange,d['intensityTrack'][sn,:], c = 'lightgreen', alpha = 0.3)
            if d['movingAvg']:
                # xRange = np.arange((d['movingAvgN']/2, (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1))
                # xRangeMA = np.arange((d['movingAvgN']/(2*d['FPS'])), (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1)/d['FPS'], 1/d['FPS'])
                xRangeMA = d['movingAvgN']/(2 * d['FPS']) + np.arange(0, (len(d['intensityTrack'][sn,:])-d['movingAvgN']+1)/d['FPS'], 1/d['FPS'])
                ax.plot(xRangeMA, moving_average(d['intensityTrack'][sn, :], d['movingAvgN']), c = 'green', label = 'DD')
            try:
                if d['frameRangePlot']:
                    d['intensityTrackDA'] = d['intensityTrackDA'][d['frameRangePlotA']:d['frameRangePlotB']]
                ax.plot(xRange, d['intensityTrackDA'][sn, :], c = 'orange', alpha = 0.3)
                if d['movingAvg']:
                    ax.plot(xRangeMA, moving_average(d['intensityTrackDA'][sn,:], d['movingAvgN']), c = 'darkorange', label = 'DA')
            except:
                pass
            try:
                if d['frameRangePlot']:
                    d['intensityTrackAA'] = d['intensityTrackAA'][d['frameRangePlotA']:d['frameRangePlotB']]
                ax.plot(xRange, d['intensityTrackAA'][sn, :], c = 'lightcoral', alpha = 0.3)
                if d['movingAvg']:
                    ax.plot(xRangeMA, moving_average(d['intensityTrackAA'][sn,:], d['movingAvgN']), c = 'red', alpha = 0.5, label = 'AA')
            except:
                pass
            ax.legend()

            ax.set_box_aspect(0.3)
            #axs[1].set_box_aspect(0.3)
            if d['showPlots']:
                plt.show()
            if d['savePlots']:
                Path(d['directorySave']+'/Intensity traces/').mkdir(parents=True, exist_ok=True)
                fig.savefig(d['directorySave']+'Intensity traces/'+d['fname']+'#'+str(sn)+' intensity track' + d['savePlotsFormat'], bbox_inches='tight')
            # fig.show()
            # fig.clf()
            ax3.cla()
            ax.cla()
            #axs[1].cla()
    plt.close(fig)
    return(d)

def plot7FRET(d):
    d['ERange'] = [-0.25, 1.25]# acceptance range for plotting
    d['SRange'] = [-0.25, 1.25]
    from imgProc import moving_average
    from pathlib import Path
    ### FRET track one by one
    print('Plotting FRET track one by one...')
    showFold = d['showFolds']
    if d['frameRangePlot']:
        d['intensityTrack'] = d['intensityTrack'][d['frameRangePlotA']:d['frameRangePlotB']]
        d['intensityTrackAA'] = d['intensityTrackAA'][d['frameRangePlotA']:d['frameRangePlotB']]
        d['intensityTrackDA'] = d['intensityTrackDA'][d['frameRangePlotA']:d['frameRangePlotB']]
    plot_E_S = 1
    if plot_E_S ==1:
        fig, axs = plt.subplots(2,1)
        ax=axs[0]
    else:
        fig, axs = plt.subplots(1,1)
        ax=axs
    for axN in fig.get_axes():
        axN.label_outer()
    plt.subplots_adjust(left=0.10,
                        bottom=0.10,
                        right=0.4,
                        top=0.4,
                        wspace=0.1,
                        hspace=0.1)
    # fig, ax = plt.figure(d['figNo'])
    d['figNo'] += 1
    # ax3 = ax.twiny()
    # ax3.set_xlabel('Time (s)')
    for sn in range(d['totSpots']):
        if 1:#(d['lifeTime'][sn] > d['intensityTrack'].shape[1] / 3):
            # ax.set_title('Spot #' + str(sn))
            # ax.set_xlabel('Frame #')
            # ax.set_xlabel('Time (s)')
            ax.set_ylabel('$E_{FRET}$') # ax = plt.gca()
            ax.set_ylim(d['ERange'][0],d['ERange'][1])
            # ax.set_xlim([0, d['intensityTrack'].shape[1]])
            ax.set_xlim([0, d['intensityTrack'].shape[1]/d['FPS']])
            # ax3.set_xlim([0, d['intensityTrack'].shape[1]/d['FPS']])
            fDD = d['intensityTrack'][sn,:]
            fAA = d['intensityTrackAA'][sn,:]
            fDA = d['intensityTrackDA'][sn,:]
            Ffret = fDA - d['FRETAlpha'] * fDD - d['FRETDelta'] * fAA
            E = Ffret / (d['FRETGamma'] * fDD + Ffret)
            if d['markEvent']:
                ax.plot((d['markEventFrame'], d['markEventFrame']),(d['ERange'][0], d['ERange'][1]))
            xRange = np.arange(0,d['intensityTrack'].shape[1]/d['FPS'],1/d['FPS'])
            ax.plot((0, d['intensityTrack'].shape[1]/d['FPS']),(0, 0),c='lightgrey', ls=':')
            ax.plot((0, d['intensityTrack'].shape[1]/d['FPS']),(1, 1),c='lightgrey', ls=':')
            ax.plot(xRange,E, c = 'lightblue')
            if d['movingAvg']:
                # xRange = np.arange((d['movingAvgN']/2, (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1))
                # xRangeMA = np.arange((d['movingAvgN']/(2*d['FPS'])), (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1)/d['FPS'], 1/d['FPS'])
                xRangeMA = d['movingAvgN']/(2 * d['FPS']) + np.arange(0, (len(E)-d['movingAvgN']+1)/d['FPS'], 1/d['FPS'])
                ax.plot(xRangeMA, moving_average(E, d['movingAvgN']), c = 'blue')
            ax.set_box_aspect(0.4)

            if plot_E_S ==1:
                ax=axs[1]
                # ax3 = ax.twiny()
                # ax3.set_xlabel('Time (s)')
                # ax.set_xlabel('Frame #')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel("$S_{FRET}$") # ax = plt.gca()
                ax.set_ylim(d['SRange'][0],d['SRange'][1])
                # ax.set_xlim([0, d['intensityTrack'].shape[1]])
                ax.set_xlim([0, d['intensityTrack'].shape[1]/d['FPS']])
                # ax3.set_xlim([0, d['intensityTrack'].shape[1]/d['FPS']])
                fDD = d['intensityTrack'][sn,:]
                fAA = d['intensityTrackAA'][sn,:]
                fDA = d['intensityTrackDA'][sn,:]
                Ffret = fDA - d['FRETAlpha'] * fDD - d['FRETDelta'] * fAA
                S = (d['FRETGamma'] * fDD + Ffret) / (d['FRETGamma'] * fDD + Ffret + fAA/d['FRETBeta'])
                if d['markEvent']:
                    ax.plot((d['markEventFrame'], d['markEventFrame']),(-0.25, 1.25))
                xRange = np.arange(0,d['intensityTrack'].shape[1]/d['FPS'],1/d['FPS'])
                ax.plot((0, d['intensityTrack'].shape[1]/d['FPS']),(0, 0),c='lightgrey', ls=':')
                ax.plot((0, d['intensityTrack'].shape[1]/d['FPS']),(1, 1),c='lightgrey', ls=':')
                ax.plot(xRange,S, c = 'wheat')
                if d['movingAvg']:
                    # xRange = np.arange((d['movingAvgN']/2, (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1))
                    # xRangeMA = np.arange((d['movingAvgN']/(2*d['FPS'])), (len(d['intensityTrack'][cc,:])-(d['movingAvgN']/2) + 1)/d['FPS'], 1/d['FPS'])
                    xRangeMA = d['movingAvgN']/(2 * d['FPS']) + np.arange(0, (len(S)-d['movingAvgN']+1)/d['FPS'], 1/d['FPS'])
                    ax.plot(xRangeMA, moving_average(S, d['movingAvgN']), c = 'orange')
                ax.set_box_aspect(0.4)
                ax=axs[0] # return ax0 for the next loop


            # axs[1].set_box_aspect(0.3)
            if d['showPlots']:
                plt.show()
            if d['savePlots']:
                Path(d['directorySave']+'/FRET traces/').mkdir(parents=True, exist_ok=True)
            fig.savefig(d['directorySave']+'FRET traces\\'+d['fname']+'#'+str(sn)+' FRET' + d['savePlotsFormat'], bbox_inches='tight')
            # fig.show()
            # fig.clf()
            if plot_E_S ==1:
                axs[0].cla()
                axs[1].cla()
            else:
                ax.cla()

            # axs[1].cla()
    plt.close(fig)
    return(d)

def plot8select(d, intensity): # to select the threshold for outlier detection of spots
    import matplotlib.pyplot as plt
    nBins = 100
    # calculate the histogram of the data by numpy
    hist, bins = np.histogram(intensity, bins=nBins)
    # Find the upper limit to avoid small number of outliers to dominate the histogram range 
    xMax = max([bins[i+1] for i in range(len(hist)) if hist[i] > max(hist)/100])
    # number a new figure
    # fig = plt.figure(1)
    fig, ax = plt.subplots(num=1)
    Color = 'green'
    if d['channelNo'] == 2:
        Color = 'red'
    plt.hist(intensity, bins=np.arange(0, xMax, np.round(xMax/nBins)), align='mid', facecolor=Color)
    ax.set_xlabel('Intensity (photon)')
    ax.set_title('Right/left click on the histogram to select the integration range, then close the window', fontsize=10)
    # initialize variables to store x-positions of selection range
    global start_pos
    global end_pos
    start_pos = None
    end_pos = None

    # define function to handle mouse clicks
    def onclick(event):
        global start_pos
        global end_pos
        
        # check if left or right mouse button was clicked
        if event.button == 1: # left-click
            start_pos = event.xdata
        elif event.button == 3: # right-click
            end_pos = event.xdata
        
        # remove previous lines
        ax.lines.pop(0) if len(ax.lines) > 0 else None
        ax.lines.pop(0) if len(ax.lines) > 0 else None
        
        # draw vertical lines for start and end positions
        if start_pos is not None:
            ax.axvline(x=start_pos, color='black', linestyle=':')
        if end_pos is not None:
            ax.axvline(x=end_pos, color='black', linestyle=':')
        
        # update the plot
        plt.draw()
    d['figHist'] = fig
    # connect the onclick function to the mouse click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # show the plot
    plt.show()
    if start_pos is not None and end_pos is not None:
        d['integMin'] = np.round(min([start_pos, end_pos]))
        d['integMax'] = np.round(max([start_pos, end_pos]))
        print('range = ' + str(d['integMin']) + ', ' + str(d['integMax']))
    # close the plot window
    plt.close()

    return(d)


def plot8(d, intensity, intensityOutlier):
    ### spot selection histogram to compare outlier spots to good spots
    fig = plt.figure(d['figNo'])
    d['figNo'] += 1
    binwidth = 10
    nBins = 100
    hist, bins , patches= plt.hist(intensityOutlier, bins=nBins)
    # Find the upper limit to avoid small number of outliers to dominate the histogram range 
    xMax = max([bins[i+1] for i in range(len(hist)) if hist[i] > max(hist)/100])
    # Clear plot
    plt.clf()    
    # n, bins, patches = plt.hist(intensity, bins=range(1, int(intensity.max() + binwidth), binwidth), align='mid', facecolor='grey' )
    plt.hist(intensity, bins=np.arange(0, xMax, np.round(xMax/nBins)), align='mid', facecolor='grey' )
    # plt.hist(intensityOutlier, bins=range(1, int(intensity.max() + binwidth), binwidth), align='mid', facecolor='lightgrey' )
    hist, bins , patches = plt.hist(intensityOutlier, bins=np.arange(0, xMax, np.round(xMax/nBins)), align='mid', facecolor='lightgrey' )
    
    plt.xlim(xmin=0, xmax=xMax)
    plt.xlabel('Intensity (photon)')
    plt.ylabel('Count')
    plt.title('Comparison of normal to outlier detected spots')
    plt.gca().set_box_aspect(1)
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave']+d['fname'] + ' spot selection histogram-ch' + str(d['channelNo']) + d['savePlotsFormat'], bbox_inches='tight')
    plt.close(fig)
    return(d)

def plot9(d):
    ### FOV with detected spots and integrating area
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    import matplotlib as mpl
    import math
    r = d['integR']
    img1 = ndimage.gaussian_filter(d['firstImgs'], sigma=d['sigmaImg']) - ndimage.gaussian_filter(d['firstImgs'], sigma=d['sigmaBkgnd'])
    img2 = np.zeros([d['imgY'],d['imgX']])
    if d['recordingMode'] == 'alex':
        img2 = ndimage.gaussian_filter(d['firstImgs2'], sigma=d['sigmaImg']) - ndimage.gaussian_filter(d['firstImgs2'], sigma=d['sigmaBkgnd'])
    img=np.asarray([img2, img1, np.zeros([d['imgY'],d['imgX']])]).transpose(1,2,0)
    img = img /(np.mean(d['SNR']) * d['firstImgsSTD'])
    ### show detected spots
    # img_adapteq = exposure.equalize_adapthist(d['firstImg']/d['firstImg'].max(), clip_limit=0.03)
    # img_eq = exposure.equalize_hist(d['firstImg']/d['firstImg'].max())
    # imgEq = 4*(d['firstImg']-d['firstImg'].min())/(d['firstImg'].max()-d['firstImg'].min())
    my_dpi = 96
    fig = plt.figure(d['figNo'], figsize=(d['imgY']/my_dpi, d['imgX']/my_dpi), dpi=my_dpi)
    d['figNo'] += 1
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img)#, vmin=0, vmax= np.mean(d['SNR']) * d['firstImgsSTD'])
    # plt.title(d['fname'])
    plt.autoscale(False)
    # fig.set_size_inches(16, 9)
    # for sn in range(d['totSpots']):
    #     ax.scatter(d['intIndex'][0, :]+d['xc'][sn], d['intIndex'][1, :]+d['yc'][sn], marker ='+', c='g', s= 0.001)
    if d['recordingMode'] == 'singleChannel':
        patches = pathAround(d['intIndex'], d['xc'], d['yc'])
        collection = PatchCollection(patches, edgecolor='lime', linewidths=0.1)
        ax.add_collection(collection)
    if d['recordingMode'] == 'singleChannelDoubleDye':
        patches = pathAround(d['intIndex'], d['xc'], d['yc'])
        collection = PatchCollection(patches, edgecolor='lime', linewidths=0.1)
        ax.add_collection(collection)
        xc2 = [x + int(d['alexSpotDist'][1]) for x in d['xc']]
        yc2 = [y + int(d['alexSpotDist'][0]) for y in d['yc']]
        patches = pathAround(d['intIndex'], xc2, yc2)
        collection = PatchCollection(patches, edgecolor='red', linewidths=0.1)
        ax.add_collection(collection)
    if (d['recordingMode'] == 'alex') or (d['recordingMode'] == 'firstFrameDifferent'):
        #     for sn in range(d['totSpots']):
        #         ax.scatter(d['intIndex2'][0, :]+d['xc'][sn]+d['alexSpotDist'][1], d['intIndex2'][1, :]+d['yc'][sn]+d['alexSpotDist'][0], marker ='x', c='r', s= 0.001)
        dist = np.linalg.norm((d['alexSpotDist'][0], d['alexSpotDist'][1])).astype(int)
        angle = math.degrees(math.atan2(d['alexSpotDist'][0], d['alexSpotDist'][1]))
        print('elongation angle=' + str(angle))
        pairs = zip(np.asarray(d['xc']), np.asarray(d['yc']))
        # patches = [FancyBboxPatch((xci, yci), dist, 0, boxstyle="round,pad="+str(r),transform=mpl.transforms.Affine2D().rotate_deg_around(*(xci,yci), angle)) for xci, yci in pairs]
        # coll = mpl.collections.PatchCollection(patches, facecolors='none', edgecolor= 'orange', linewidths=0.1 )
        # plt.gca().add_collection(coll)
        ### add a path patch
        patches = pathAround(d['intIndex'], d['xc'], d['yc'])
        collection = PatchCollection(patches, edgecolor='green', linewidths=0.1)
        ax.add_collection(collection)
        patches = pathAround(d['intIndex2'], d['xc2'], d['yc2'])
        collection = PatchCollection(patches, edgecolor='red', linewidths=0.1)
        ax.add_collection(collection)
    try: #to show the masks positions from MT channel
        xc = np.squeeze(d['maskList'][:, 1])
        yc = np.squeeze(d['maskList'][:, 0])
        patches = pathAround(d['maskIndexCircle'], xc, yc)
        collection = PatchCollection(patches, edgecolor='grey', linewidths=0.1)
        ax.add_collection(collection)
    except:
        pass

    plt.rcParams['pdf.fonttype'] = 42
    if d['showPlots']:
        plt.show()
    if d['savePlots']:
        plt.savefig(d['directorySave'] + d['fname'] + ' peaks positions' + d['savePlotsFormat'], dpi=my_dpi, bbox_inches='tight')#, backend='pgf') # pgf: reduce filesize, may increase execution time
    plt.close(fig)
    return(d)


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# import winsound
from FRETcalc import *
from HMM import *

def HMMplot(d):
    sn = 20
    trace = d['intensityTrackDA'][sn, :]
    trace_means = np.array([400, 600])
    run_hmm(trace, 1, fig=None, tr_means=trace_means, dir_=d['directorySave'], sigma=0.05, numstates=2, freq=1, frames = 119, save=True, sequence=1)

def customCMap(d): #color map for plot
    origCMap = cm.get_cmap('gray', 256)
    newCMap = origCMap(np.linspace(0.6, 1, 256))
    if d['plot5CMap'] == 'gray':
        d['newCMap'] = ListedColormap(newCMap)
    else:
        d['newCMap'] = 'viridis'
    return(d)