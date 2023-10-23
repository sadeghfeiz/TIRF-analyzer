import numpy as np
from plotter import *
import scipy.ndimage.filters as scfilters
import scipy.ndimage as ndimage
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.collections
import tifffile
# from skimage import exposure
from scipy.optimize import curve_fit
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
import pandas as pd

def FRETcalc(d):
    print('FRET calculations...')
    totSpots = d['totSpots']
    d['ERange'] = [-0.25, 1.25]# acceptance range for calculations
    d['SRange'] = [-0.25, 1.25]
    frameRange=[0, d['totFrames']]# accumulation frame range
    negativeAsOutlier = 0 # remove spots with negative values (less than background)
    correctForGamma = 0
    d['fit2DGaussian'] = 1
    d['FRETn'] = 4#GUI
    d['correctOn2DHist'] = 1 # fit to 2D histogram instead of directly fitting on scatter
    fDDframe = []
    fDAframe = []
    fAAframe = []
    fDD = []
    fDA = []
    fAA = []
    # initial values to avoid error when saving to .xls file:
    d['muE00'] = [0, 1]
    d['muS11'] = [1, 0]
    d['muE10'] = [1, 0] #fit centers for E on 2D plot
    d['muS10'] = [0, 1] #fit centers for E on 2D plot
    d['FRETAlpha'] = 0
    d['FRETDelta'] = 0
    d['FRETGamma'] = 1
    d['FRETBeta'] = 1
    for fn in range(frameRange[0],frameRange[1]):
        for sn in range(d['totSpots']):
            if (d['lifeTime2'][sn] >= fn) or (d['lifeTime'][sn] >= fn): #assign non-bleached spots in the frame to calculations
                fDDframe = np.append(fDDframe,d['intensityTrack'][sn,fn])
                fDAframe = np.append(fDAframe,d['intensityTrackDA'][sn,fn])
                fAAframe = np.append(fAAframe,d['intensityTrackAA'][sn,fn])
        if negativeAsOutlier:#delete spot from calculations if it's value is negative
            deletedObj = 0
            for ns in range(totSpots):
                nsRev = totSpots-ns -1
                if (fDDframe[nsRev] < 0) or (fDAframe[nsRev] < 0) or (fAAframe[nsRev] < 0):
                    fDDframe = np.delete(fDDframe, nsRev, 0)
                    fDAframe = np.delete(fDAframe, nsRev, 0)
                    fAAframe = np.delete(fAAframe, nsRev, 0)
                    deletedObj += 1
            print('deletedObj: ', deletedObj)
            totSpots = totSpots - deletedObj
        # Accumulate fDD, fDA and fAA over different frames:
        fDD = np.concatenate((fDD,fDDframe), axis = 0)
        fDA = np.concatenate((fDA,fDAframe), axis = 0)
        fAA = np.concatenate((fAA,fAAframe), axis = 0)
    # Raw SE calculation:
    print('working on raw data...')
    Sraw = (fDD + fDA)/(fDD + fDA + fAA)
    Eraw = fDA / (fDD + fDA)
    ###  plots
    d = SEplot(d, E=Eraw, S=Sraw, corrType='RawData', plotType='scatter')
    d = SEplot(d, E=Eraw, S=Sraw, corrType='RawData', plotType='2Dhist')

    ### Cross correction #############################################################################################
    print('working on cross correction...')
    ### define A-only and D-only
    EDonly = d['muE10'][-1]
    SAonly = d['muS11'][1]
    # print ('ED-only',EDonly)
    # print ('SA-only',SAonly)
    alpha = EDonly / (1 - EDonly)
    delta = SAonly / (1 - SAonly)
    d['FRETAlpha'] = alpha
    d['FRETDelta'] = delta
    ### cross corrected S and E:
    Ffret = fDA - alpha * fDD - delta * fAA
    E = Ffret / (fDD + Ffret)
    S = (fDD + Ffret) / (fDD + Ffret + fAA)

    ### cross corrected plot
    d = SEplot(d, E=E, S=S, corrType='CrossCorr', plotType='scatter')
    d = SEplot(d, E=E, S=S, corrType='CrossCorr', plotType='2Dhist')

    ### Gamma correction ###############################################################################################
    print('working on Gamma correction...')
    ### Gamma and Beta constants:
    S1, S2 = d['muS10'][1:3]
    E1, E2 = d['muE10'][1:3]
    m = (1/S2-1/S1)/(E2 - E1)
    b = 1/S1 - m * E1
    d['FRETGamma'] = (b - 1)/(m + b - 1)
    d['FRETBeta'] = m + b - 1
    gamma = d['FRETGamma']
    beta = d['FRETBeta']
    print(f'Gamma: {gamma}')
    print(f'Beta: {beta}')
    ### Gamma corrected S and E:
    Egamma = Ffret / (gamma * fDD + Ffret)
    Sgamma = (gamma * fDD + Ffret) / (gamma * fDD + Ffret + fAA/beta)
    ### Gamma corrected plots:
    d = SEplot(d, E=Egamma, S=Sgamma, corrType='GammaCorr', plotType='scatter')
    d = SEplot(d, E=Egamma, S=Sgamma, corrType='GammaCorr', plotType='2Dhist')

    ### save constants to excel file
    writer = pd.ExcelWriter(d['directorySave']+d['fname']+'.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df1 = pd.DataFrame([['FRETalpha', d['FRETAlpha']], ['FRETDelta', d['FRETDelta']], ['FRETGamma', d['FRETGamma']], ['FRETBeta', d['FRETBeta']]])
    df1.to_excel(writer, sheet_name='parameters', startrow=writer.sheets['parameters'].max_row, index=False, header=False)
    df2 = pd.DataFrame(np.array([['muE00', d['muE00']], ['muS11', d['muS11']], ['muE10', d['muE10']], ['muS10', d['muS10']]]))
    df2.to_excel(writer, sheet_name='parameters', startrow=writer.sheets['parameters'].max_row, index=False, header=False)
    df3 = pd.DataFrame([['ERange', d['ERange']], ['SRange', d['SRange']]])
    df3.to_excel(writer, sheet_name='parameters', startrow=writer.sheets['parameters'].max_row, index=False, header=False)

    writer.save()
    return d

def SEplot(d, E, S, corrType, plotType):
    color = 'gray'
    if corrType == 'RawData': color ='skyblue'
    if corrType == 'CrossCorr': color ='blue'
    if corrType == 'GammaCorr': color ='navy'
    w = 9
    h = 9
    nBins = 150
    numberOfPopulations = d['FRETn']
    ERange = d['ERange'] # acceptance range for calculations
    SRange = d['SRange']
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [4, 1],'height_ratios': [1, 4]})
    fig.set_size_inches(w, h)
    SinRange = []
    EinRange = []
    for sn in range(len(E)):
        if (ERange[0] < E[sn] < ERange[1]) and (SRange[0] < S[sn] < SRange[1]):
            SinRange.append(S[sn])
            EinRange.append(E[sn])
    yHist00, xHist00, _ = ax[0, 0].hist(EinRange, bins=nBins, range=ERange, align='mid', facecolor=color, alpha=0.5, label = corrType)
    yHist11, xHist11, _ = ax[1, 1].hist(SinRange, bins=nBins, range=SRange, align='mid', facecolor=color, alpha=0.5, orientation='horizontal')

    try:
        ax, d['muE00'] = plotFits(yHist00, xHist00, numberOfPopulations, ax, 0, color)
    except:
        pass
    try:
        ax, d['muS11'] = plotFits(yHist11, xHist11, numberOfPopulations, ax, 1, color)
    except:
        pass

    ### plot scatter/2D hist
    if plotType == 'scatter':
        ax[1, 0].scatter(E, S, marker = '.', s=1, c= color, alpha=0.05)
        # try:### because fitting is not always successful
        d = GMM(d, E, S, ax[1, 0]) # fit Gaussian Mixture Model
        # except:
        #     pass
    if plotType == '2Dhist':
        ax[1, 0].hexbin(E,S, gridsize=400, mincnt=int(np.max(yHist00)/50), bins = 'log' , extent = [-5, 5, -5, 5])#, extent=[ERange[0], ERange[1], SRange[0], SRange[1]])
        try:### because fitting is not always successful
            d = Gaussian2DFit(d, E, S, ax[1, 0])
        except:
            pass

    ax[0, 0].set_xlim(ERange)
    ax[0, 1].axis('off')
    ax[1, 0].set_xlim(ERange)
    ax[1, 0].set_ylim(SRange)
    ax[1, 0].set(xlabel='E', ylabel='S')
    # ax[1, 0].set_yticks([0, 1], minor=False)
    ax[1, 0].get_xticklabels()[-1].set_color("green")
    ax[1, 0].get_yticklabels()[-1].set_color("green")
    ax[1, 0].axhline(0, color='k', linewidth=1)
    ax[1, 0].axhline(1, color='k', linewidth=1)
    ax[1, 0].axvline(0, color='k', linewidth=1)
    ax[1, 0].axvline(1, color='k', linewidth=1)
    ax[1, 1].set_ylim(SRange)
    ax[1, 1].set_xlim(left = 0)
    for axN in fig.get_axes():
        axN.label_outer()
    fig.tight_layout()
    handles, labels = ax[0,0].get_legend_handles_labels()
    ax[0, 1].legend(handles, labels, loc='upper center', markerscale=8)
    plt.savefig(d['directorySave'] + d['fname'] + ' ' + corrType + ' ' + plotType + ' SE plot.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    return(d)

def fitHuber(x,y): #linear fit removing outliers
    epsilon = 1 # huber loss
    from sklearn.linear_model import HuberRegressor, Ridge
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    x = np.vstack(x)
    huber.fit(x, y)
    # plt.plot(x, coef_,"r-", label="huber loss, %s" % epsilon)
    return huber

# define 1D Gaussian fits
def Gauss1D(x, mu, sigma, A):
    return abs(A)*np.exp(-(x-mu)**2/2/sigma**2)
def tetramodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4):
    return Gauss1D(x,mu1,sigma1,A1)+Gauss1D(x,mu2,sigma2,A2)+Gauss1D(x,mu3,sigma3,A3)+Gauss1D(x,mu4,sigma4,A4)
def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    return Gauss1D(x,mu1,sigma1,A1)+Gauss1D(x,mu2,sigma2,A2)+Gauss1D(x,mu3,sigma3,A3)
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return Gauss1D(x,mu1,sigma1,A1)+Gauss1D(x,mu2,sigma2,A2)
def plotFits(yHist, xHist, numberOfPopulations, ax, axNo, color):
    mu = []# fit centers
    if axNo == 0:
        #parameters: mu, sigma, Amp (D-only, A-only, FRET pop1, FREt pop2)
        paramBounds = ([-0.1, 0.01,   0,       0.2, 0.3,    0 ,       0,   0.01,   0,        0,  0.01, 0],
                       [0.25, 0.2, np.inf,     0.8,  5, np.inf,       1.1, 0.3, np.inf,      1.1, 0.3, np.inf])
        expected = (0.1, 0.1, yHist.max(),    0.5, 1, yHist.max(),     0.5 , 0.1, yHist.max(),        0.1, 0.1, yHist.max())
    if axNo == 1:
        maxForDonly = yHist[100:].max()
        maxForAonly = 2 * yHist[:50].max()
        #parameters: mu, sigma, Amp
        ###                  D-only,                    A-only,                        FREt pop)
        paramBounds = ([0.8, 0.01, 0,              -0.1, 0.01, 0,                   0.2, 0.01, 0],
                       [1.2, 0.2,  np.inf,          0.1, 0.15,  np.inf,             1,   0.3,  np.inf])
        expected =     (1,   0.07, maxForDonly,     0.0, 0.1,  maxForAonly,         0.5, 0.1,  yHist.max())
    x_fit = np.linspace(xHist.min(), xHist.max(), 500)
    if numberOfPopulations == 1:
        params, cov = curve_fit(Gauss1D, xHist[:-1], yHist, expected[:3])
        ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params), color=color, lw=1)
        ax[axNo, axNo].plot([[params[0]-params[1], 10], [params[0]+params[1], 10]], color=color)
        mu.append(params[0])
        sigma=np.sqrt(np.diag(cov))
        if axNo == 0:
            ax[0, 0].axvline(params[0], linestyle='--', linewidth=1, color=color)
            ax[1, 0].axvline(params[0], linestyle='--', linewidth=1, color=color)
    if numberOfPopulations == 2:
        params, cov = curve_fit(bimodal, xHist[:-1], yHist, p0 = expected[:6], bounds=(paramBounds[0][:6], paramBounds[1][:6]))
        ax[axNo, axNo].plot(x_fit, bimodal(x_fit, *params[:6]), color=color, lw=1)
        #...and individual Gauss curves
        ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[:3]), color=color, lw=1, ls="--", label='$x_0$='+str(params[0].round(3))+' $\sigma$='+str(params[1].round(3)))
        ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[3:]), color=color, lw=1, ls="--", label='$x_0$='+str(params[3].round(3))+' $\sigma$='+str(params[4].round(3)))
        mu.append(params[0])
        mu.append(params[6])
        sigma=np.sqrt(np.diag(cov))
        if axNo == 0:
            ax[0, 0].axvline(params[0], linestyle='--', linewidth=1,color=color)
            ax[0, 0].axvline(params[6], linestyle='--', linewidth=1,color=color)
            ax[1, 0].axvline(params[0], linestyle='--', linewidth=1,color=color)
            ax[1, 0].axvline(params[6], linestyle='--', linewidth=1,color=color)
    if numberOfPopulations == 3:
        params, cov = curve_fit(trimodal, xHist[:-1], yHist, p0 = expected, bounds=(paramBounds[0][:9], paramBounds[1][:9]))
        if axNo == 0:
            ax[0, 0].axvline(params[0], linestyle='--', linewidth=1,color=color)
            ax[0, 0].axvline(params[6], linestyle='--', linewidth=1,color=color)
            # ax[0, 0].axvline(params[6], linestyle='--', linewidth=1,color=color)
            ax[axNo, axNo].plot(x_fit, trimodal(x_fit, *params), color=color, lw=1)
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[:3]), color=color, lw=1, ls=":")#, label='$x_0$='+str(params[0].round(3))+' $\sigma$='+str(params[1].round(3)))
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[3:6]), color=color, lw=1, ls=":")#, label='$x_0$='+str(params[3].round(3))+' $\sigma$='+str(params[4].round(3)))
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[6:]), color=color, lw=1, ls=":")#, label='$x_0$='+str(params[6].round(3))+' $\sigma$='+str(params[7].round(3)))
        if axNo == 1:
            # ax[1, 1].axvline(params[0], linestyle='--', linewidth=1,color=color)
            ax[1, 1].axhline(params[3], linestyle='--', linewidth=1,color=color)
            ax[1, 1].axhline(params[6], linestyle='--', linewidth=1,color=color)
            ax[axNo, axNo].plot(trimodal(x_fit, *params), x_fit, color=color, lw=1)
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[:3]),x_fit, color=color, lw=1, ls=":")#, label='$x_0$='+str(params[0].round(3))+' $\sigma$='+str(params[1].round(3)))
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[3:6]),x_fit, color=color, lw=1, ls=":")#, label='$x_0$='+str(params[3].round(3))+' $\sigma$='+str(params[4].round(3)))
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[6:]),x_fit, color=color, lw=1, ls=":")#, label='$x_0$='+str(params[6].round(3))+' $\sigma$='+str(params[7].round(3)))
    if numberOfPopulations == 4:

        if axNo == 0:
            params, cov = curve_fit(tetramodal, xHist[:-1], yHist, p0 = expected, bounds=(paramBounds[0][:12], paramBounds[1][:12]))
            mu.append(params[0])
            mu.append(params[6])
            mu.append(params[9])
            ax[0, 0].axvline(params[0], linestyle='--', linewidth=1,color='green')
            ax[0, 0].axvline(params[6], linestyle='--', linewidth=1,color=color)
            ax[0, 0].axvline(params[9], linestyle='--', linewidth=1,color=color)
            ax[0, 0].set_xticks(mu, minor=False)
            ax[0, 0].set_xticklabels(np.round(mu,2))
            ax[0, 0].xaxis.tick_top()
            ax[0, 0].get_xticklabels()[0].set_color('green')
            ax[axNo, axNo].plot(x_fit, tetramodal(x_fit, *params), color=color, lw=1)
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[:3]), color='green', lw=1, ls=":")#, label='$x_0$='+str(params[0].round(3))+' $\sigma$='+str(params[1].round(3)))
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[3:6]), color=color, lw=1, ls=":")#, label='$x_0$='+str(params[3].round(3))+' $\sigma$='+str(params[4].round(3)))
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[6:9]), color=color, lw=1, ls=":")#, label='$x_0$='+str(params[6].round(3))+' $\sigma$='+str(params[7].round(3)))
            ax[axNo, axNo].plot(x_fit, Gauss1D(x_fit, *params[9:12]), color=color, lw=1, ls=":")
        if axNo == 1:
            max1= np.argmax(yHist[0:25])
            max2= np.argmax(yHist[25:45])+1+25
            xHistSlice=np.delete(xHist,np.array(range(max1,max2)))
            yHistSlice=np.delete(yHist,np.array(range(max1,max2)))
            yHistSliceReverse = yHist
            yHistSliceReverse[0:max1] = 0
            yHistSliceReverse[max2:] = 0
            nBins = 150
            SRange = [-0.25, 1.25]
            ax[1,1].barh(np.arange(-0.25,1.25,0.01),height=0.01, width=yHistSliceReverse, color='lightgrey', align = 'edge')
            params, cov = curve_fit(trimodal, xHistSlice[:-1], yHistSlice, p0 = expected, bounds=(paramBounds[0][:9], paramBounds[1][:9]))
            mu.append(params[0])
            mu.append(params[3])
            mu.append(params[6])
            ax[1, 1].axhline(params[0], linestyle='--', linewidth=1, color='green')
            ax[1, 1].axhline(params[3], linestyle='--', linewidth=1, color='red')
            ax[1, 1].axhline(params[6], linestyle='--', linewidth=1, color=color)
            ax[1, 1].set_yticks(mu, minor=False)
            ax[1, 1].set_yticklabels(np.round(mu,2))
            ax[1, 1].yaxis.tick_right()
            ax[0, 0].get_xticklabels()[0].set_color('green')
            ax[1, 1].get_yticklabels()[0].set_color('green')
            ax[1, 1].get_yticklabels()[1].set_color('red')
            ax[axNo, axNo].plot(trimodal(x_fit, *params), x_fit, color=color, lw=1)
            # ax[axNo, axNo].plot(tetramodal(x_fit, *params), x_fit, color=color, lw=1)
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[:3]),x_fit, color='green', lw=1, ls=":")#, label='$x_0$='+str(params[0].round(3))+' $\sigma$='+str(params[1].round(3)))
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[3:6]),x_fit, color='red', lw=1, ls=":")#, label='$x_0$='+str(params[3].round(3))+' $\sigma$='+str(params[4].round(3)))
            ax[axNo, axNo].plot(Gauss1D(x_fit, *params[6:9]),x_fit, color=color, lw=1, ls=":")#, label='$x_0$='+str(params[6].round(3))+' $\sigma$='+str(params[7].round(3)))
            # ax[axNo, axNo].plot(Gauss1D(x_fit, *params[9:12]),x_fit, color=color, lw=1, ls=":")#, label='$x_0$='+str(params[6].round(3))+' $\sigma$='+str(params[7].round(3)))

    return(ax, mu)


def Gauss2D(tup, amplitude, xo, yo, sigma_x, sigma_y, theta):
    (x , y) = tup
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( -(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()
def Gaussian2Dbimodal(tup, a1, xo1, yo1, sx1, sy1, t1, a2, xo2, yo2, sx2, sy2, t2):
    return Gauss2D(tup, a1, xo1, yo1, sx1, sy1, t1) + Gauss2D(tup, a2, xo2, yo2, sx2, sy2, t2)
def Gaussian2Dtrimodal(tup, a1, xo1, yo1, sx1, sy1, t1, a2, xo2, yo2, sx2, sy2, t2, a3, xo3, yo3, sx3, sy3, t3):
    return Gauss2D(tup, a1, xo1, yo1, sx1, sy1, t1) + Gauss2D(tup, a2, xo2, yo2, sx2, sy2, t2) + Gauss2D(tup, a3, xo3, yo3, sx3, sy3, t3)
def Gaussian2Dtetramodal(tup, a1, xo1, yo1, sx1, sy1, t1, a2, xo2, yo2, sx2, sy2, t2, a3, xo3, yo3, sx3, sy3, t3, a4, xo4, yo4, sx4, sy4, t4):
    return Gauss2D(tup, a1, xo1, yo1, sx1, sy1, t1) + Gauss2D(tup, a2, xo2, yo2, sx2, sy2, t2) + Gauss2D(tup, a3, xo3, yo3, sx3, sy3, t3)+ Gauss2D(tup, a4, xo4, yo4, sx4, sy4, t4)

def Gaussian2DFit(d, E, S, ax):
    numberOfPopulations = d['FRETn']
    nBins = 150
    import scipy.optimize as opt
    # Create x and y indices
    x = np.linspace(d['ERange'][0], d['ERange'][1], nBins)
    y = np.linspace(d['SRange'][0], d['SRange'][1], nBins)
    x, y = np.meshgrid(x, y)
    tup = (x, y)
    xls = np.linspace(d['ERange'][0], d['ERange'][1], nBins+1)
    yls = np.linspace(d['SRange'][0], d['SRange'][1], nBins+1)
    hist, xedges, yedges = np.histogram2d(S, E, bins=(xls, yls))
    inf = np.inf
    amp1 = np.max(hist)
    # Guess:      amplitude, xo, yo,sigmaX,sigmaY,theta
    ###                           A-only                                 pop 1                                      pop 2                                       D-only
    initial_guess = (amp1, 0.5, -0.1, 0.6, 0.05, 0,         amp1/2,  0.5, 0.5, 0.1,  0.1,   0,       amp1/2, 0.1, 0.5, 0.1,  0.1,  -0.,            amp1, 0.1, 1.0, 0.1,  0.1, 0,)
    paramBoundsMin =[0,    0.2, -0.2, 0.1, 0.01,-0.05,         0,    0,   0.1, 0.01, 0.01, -4,         0,    0,   0.1, 0.01, 0.01, -4, 0,          -0.1, 0.7, 0.01, 0.01,-4,]
    paramBoundsMax =[inf,  0.8,  0.3, 5.0, 0.1 , 0.05,        inf,   1,   0.8, 0.5,  0.5,   4,        inf,   1,   0.8, 0.5,  0.5,   4,              inf,  0.3, 1.2, 0.3,  0.3, 4,]
    paramBounds = (paramBoundsMin, paramBoundsMax)

    popt =[]
    if numberOfPopulations == 1:
        popt, pcov = opt.curve_fit(Gauss2D, tup, hist.reshape(nBins**2), p0=initial_guess[:6])
        data_fitted = Gauss2D(tup, *popt)
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 8, colors='b')
    if numberOfPopulations == 2:
        popt, pcov = opt.curve_fit(Gaussian2Dbimodal, tup, hist.reshape(nBins**2), p0=initial_guess[:12])
        data_fitted = Gauss2D(tup, *popt[:6])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 8, colors='b')
        data_fitted = Gauss2D(tup, *popt[6:])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 8, colors='b')
    if numberOfPopulations == 3:
        popt, pcov = opt.curve_fit(Gaussian2Dtrimodal, tup, hist.reshape(nBins**2), p0=initial_guess[:18], bounds=paramBounds)
        data_fitted = Gauss2D(tup, *popt[:6])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='c')
        data_fitted = Gauss2D(tup, *popt[6:12])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='c')
        data_fitted = Gauss2D(tup, *popt[12:18])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='c')
    if numberOfPopulations == 4:
        popt, pcov = opt.curve_fit(Gaussian2Dtetramodal, tup, hist.reshape(nBins**2), p0=initial_guess[:24], bounds=paramBounds)
        data_fitted = Gauss2D(tup, *popt[:6])
        # ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='green')
        data_fitted = Gauss2D(tup, *popt[6:12])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='orange')
        data_fitted = Gauss2D(tup, *popt[12:18])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='orange')
        data_fitted = Gauss2D(tup, *popt[18:24])
        ax.contour(x, y, data_fitted.reshape(nBins, nBins), 3, colors='orange')
    muE10 = [popt[1], popt[7],popt[13],popt[19]]
    muS10 = [popt[2], popt[8],popt[14],popt[20]]

    ax.scatter(muE10[1:3], muS10[1:3], marker = '.', s=5, c='orange')
    ax.scatter(muE10[-1], muS10[-1], marker = '.', s=5, c='orange')
    ax.set_xticks(muE10[1:], minor=False)
    ax.set_yticks(muS10[1:], minor=False)
    ax.set_yticklabels(np.round(muS10[1:],2))
    ax.set_xticklabels(np.round(muE10[1:],2), rotation=45, ha='right')
    if d['correctOn2DHist']:
        d['muE10'] = muE10
        d['muS10'] = muS10
    # print('fit parameters: ', popt)

    return (d)




def GMM(d, E, S, ax):#Gaussian mixture models (NOT used anymore!)
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm, multivariate_normal
    from matplotlib.colors import LogNorm
    from sklearn import mixture
    ERange = d['ERange']
    SRange = d['SRange']
    nBins = 150
    S2=[]
    E2=[]
    for sn in range(len(E)): #select spots in range
        if (ERange[0] < E[sn] < ERange[1]) and (SRange[0] < S[sn] < SRange[1]):
            S2.append(S[sn])
            E2.append(E[sn])
    # concatenate the two datasets into the final training set:
    X_train = np.vstack([E2, S2]).T
    # fit a Gaussian Mixture Model with two components
    # clf = mixture.GaussianMixture(n_components=d['FRETn'], covariance_type="full", tol=5*1e-1)
    clf = mixture.BayesianGaussianMixture(n_components=d['FRETn'], covariance_type="full", tol=1*1e-3, max_iter = 200)
    clf.fit(X_train)
    ## sort fit peaks positions
    sortlist=clf.means_[:,1].argsort()
    mu = clf.means_.tolist()
    mu.sort(key=lambda x:x[1])
    mu = np.array(mu)
    muE10 = mu[:, 0]
    muS10 = mu[:, 1]
    # display predicted scores by the model as a contour plot
    x = np.linspace(ERange[0], ERange[1], nBins)
    y = np.linspace(SRange[0], SRange[1], nBins)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    k = clf.means_.shape[0]
    for i in sortlist[1:]: #avoid A-only populations
        mean = clf.means_[i]
        cov = clf.covariances_[i]
        Z = multivariate_normal(mean, cov).pdf(XX).reshape(X.shape)
        ax.contour(X, Y, Z, 3, colors = 'orange')
    ax.scatter(muE10[1:3],muS10[1:3], c= 'orange', s=5)
    ax.scatter(muE10[-1],muS10[-1], c= 'orange', s=5)
    ax.set_xticks(muE10[1:], minor=False)
    ax.set_yticks(muS10[1:], minor=False)
    ax.set_yticklabels(np.round(muS10[1:],2))
    ax.set_xticklabels(np.round(muE10[1:],2), rotation=45, ha='right')

    # plt.title("likelihood predicted by a GMM")
    ax.axis("tight")
    # ax.show()
    if d['correctOn2DHist'] == 0:
        d['muE10'] = muE10
        d['muS10'] = muS10
    return(d)


