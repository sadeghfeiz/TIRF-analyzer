from imgProc import *
import pandas as pd
import numpy as np
import traceback

def saver(d):
    directory = d['directorySave']
    fname = d['fname']
    intensityTrack = d['intensityTrack']
    writer = pd.ExcelWriter(directory + fname +'.xlsx', engine='xlsxwriter')
    df0 = pd.DataFrame.from_dict(d, orient='index', columns=['Value'])
    df0.to_excel(writer, sheet_name='parameters')
    df1 = pd.DataFrame(intensityTrack)
    df1.to_excel(writer, sheet_name='intensityTrack')
    try:
        df1DA = pd.DataFrame(d['intensityTrackDA'])
        df1DA.to_excel(writer, sheet_name='intensityTrackDA')
    except:
        pass
    if d['recordingMode'] == 'alex':
        df1AA = pd.DataFrame(d['intensityTrackAA'])
        df1AA.to_excel(writer, sheet_name='intensityTrackAA')
        spotStat = np.concatenate(([d['xc']], [d['yc']], [d['lifeTime']], [d['accuPhoton']], [d['lifeTime2']], [d['accuPhoton2']], [d['spotLabel']]), axis=0)
        spotStatHeaders = ['xc', 'yc', 'lifeTime', 'accuPhoton', 'lifeTime2', 'accuPhoton2', 'label']
        frameStat = np.concatenate(([d['Iavg']], [d['nSpots']], [d['Iavg2']], [d['nSpots2']]), axis=0)
        frameStatHeaders = ['Iavg', 'nSpots', 'Iavg2', 'nSpots2']
    else:
        spotStat = np.concatenate(([d['xc']], [d['yc']], [d['lifeTime']], [d['accuPhoton']]), axis=0)
        spotStatHeaders = ['xc', 'yc', 'lifeTime', 'accuPhoton']
        frameStat = np.concatenate(([d['Iavg']], [d['nSpots']]), axis=0)
        frameStatHeaders = ['Iavg', 'nSpots']
    df2 = pd.DataFrame(spotStat.T)
    df2.columns = spotStatHeaders
    df2.to_excel(writer, sheet_name='spotStat')
    df3 = pd.DataFrame(frameStat.T)
    df3.columns = frameStatHeaders
    df3.to_excel(writer, sheet_name='frameStat')

    try:
        dfD = pd.DataFrame(d['DonlyList'])
        dfD.to_excel(writer, sheet_name='DonlyList')
        dfA = pd.DataFrame(d['AonlyList'])
        dfA.to_excel(writer, sheet_name='AonlyList')
        dfA = pd.DataFrame(d['pairList'])
        dfA.to_excel(writer, sheet_name='pairList')
    except:
        pass
    writer.save()
    return ()


def loader(directory, fname):
    df0 = pd.read_excel(directory + fname + '.xlsx', sheet_name='parameters', index_col=0)
    # dl : dictionary loaded, to prevent interference with the new generated d as it will be saved again
    dl = df0.T.to_dict('records')[0] #'dict', list, 'series', 'split', 'records', 'index'
    dl['fname'] = fname #to correct if the file name is changed and is different from saved parameter
    dl['directorySource'] = directory #to correct if the file location is changed and is different from saved parameter

    for key, value in dl.items(): #convert array from string format to numbers
        try:
            dl[key] = eval(value)
        except: pass
    df1 = pd.read_excel(directory + fname +'.xlsx', sheet_name='intensityTrack', index_col=0)
    dl['intensityTrack'] = pd.DataFrame.to_numpy(df1)
    df2 = pd.read_excel(directory + fname +'.xlsx', sheet_name='spotStat', index_col=0)
    spotStat = pd.DataFrame.to_numpy(df2.T)
    df3 = pd.read_excel(directory + fname +'.xlsx', sheet_name='frameStat', index_col=0)
    frameStat = pd.DataFrame.to_numpy(df3.T)
    dl['xc'] = spotStat[0,:]
    dl['yc'] = spotStat[1,:]
    dl['lifeTime'] = spotStat[2,:]
    dl['accuPhoton'] = spotStat[3,:]
    dl['Iavg'] = frameStat[0,:]
    dl['nSpots'] = frameStat[1,:]
    if dl['recordingMode'] == 'singleChannelDoubleDye':
        df1DA = pd.read_excel(directory + fname +'.xlsx', sheet_name='intensityTrackDA', index_col=0)
        dl['intensityTrackDA'] = pd.DataFrame.to_numpy(df1DA)
    if dl['recordingMode'] == 'alex':
        dl['lifeTime2'] = spotStat[4,:]
        dl['accuPhoton2'] = spotStat[5,:]
        dl['spotLabel'] = spotStat[6,:]
        dl['Iavg2'] = frameStat[2,:]
        dl['nSpots2'] = frameStat[3,:]
        df1DA = pd.read_excel(directory + fname +'.xlsx', sheet_name='intensityTrackDA', index_col=0)
        dl['intensityTrackDA'] = pd.DataFrame.to_numpy(df1DA)
        df1AA = pd.read_excel(directory + fname +'.xlsx', sheet_name='intensityTrackAA', index_col=0)
        dl['intensityTrackAA'] = pd.DataFrame.to_numpy(df1AA)
    return dl

def getSettings(ui, d):
    try: d['saveOnDifferentFolder'] = ui.saveFolder.isChecked()
    except Exception as err:print (err)
    try: d['directoryAlt'] = ui.saveFolderField.text()
    except Exception as err:print (err)
    try: d['recordingModeIndex'] = ui.recordingMode.currentIndex()
    except Exception as err:print (err)
    try: d['FPS'] = ui.FPSValue.value()
    except Exception as err:print (err)
    try: d['ADfactor'] = ui.ADfactorValue.value()
    except Exception as err:print (err)
    try: d['tempBin'] = ui.tempBinning.isChecked()
    except Exception as err:print (err)
    try: d['spBin'] = ui.spBinning.isChecked()
    except Exception as err:print (err)
    try: d['crop'] = ui.crop.isChecked()
    except Exception as err:print (err)
    try: d['cropXA'] = ui.cropXA.value()
    except Exception as err:print (err)
    try: d['cropXB'] = ui.cropXB.value()
    except Exception as err:print (err)
    try: d['cropYA'] = ui.cropYA.value()
    except Exception as err:print (err)
    try: d['cropYB'] = ui.cropYB.value()
    except Exception as err:print (err)
    try: d['LimitFrameRange'] = ui.frameRange.isChecked()
    except Exception as err:print (err)
    try: d['frameRangeA'] = ui.frameRangeA.value()
    except Exception as err:print (err)
    try: d['frameRangeB'] = ui.frameRangeB.value()
    except Exception as err:print (err)
    try: d['blurImg'] = ui.blurImg.isChecked()
    except Exception as err:print (err)
    try: d['sigmaImg'] = ui.blurImgValue.value()
    except Exception as err:print (err)
    try: d['removeBkgnd'] = ui.removeBkgnd.isChecked()
    except Exception as err:print (err)
    try: d['sigmaBkgnd'] = ui.bkgndFiltValue.value()
    except Exception as err:print (err)
    try: d['realBkgnd'] = ui.removeSpotsOfBkgnd.isChecked()
    except Exception as err:print (err)

    try: d['FFS'] = ui.avgForSpotDetectValue1.value()
    except Exception as err:print (err)
    try: d['seekR'] = ui.seekingRvalue1.value()
    except Exception as err:print (err)
    try: d['blinking'] = ui.ignorBlinking1.isChecked()
    except Exception as err:print (err)
    try: d['blinkDuration'] = ui.ignorBlinkingValue1.value()
    except Exception as err:print (err)
    try: d['QE'] = ui.QEvalue1.value()
    except Exception as err:print (err)
    try: d['integrationModeIndex1'] = ui.integrationMode1.currentIndex()
    except Exception as err:print (err)
    try: d['integR'] = ui.integrationValue1.value()
    except Exception as err:print (err)
    try: d['integMin'] = ui.acceptanceIntA1.value()
    except Exception as err:print (err)
    try: d['integMax'] = ui.acceptanceIntB1.value()
    except Exception as err:print (err)

    try: d['FFS2'] = ui.avgForSpotDetectValue2.value()
    except Exception as err:print (err)
    try: d['seekR2'] = ui.seekingRvalue2.value()
    except Exception as err:print (err)
    try: d['blinking2'] = ui.ignorBlinking2.isChecked()
    except Exception as err:print (err)
    try: d['blinkDuration2'] = ui.ignorBlinkingValue2.value()
    except Exception as err:print (err)
    try: d['QE2'] = ui.QEvalue2.value()
    except Exception as err:print (err)
    try: d['integrationModeIndex2'] = ui.integrationMode2.currentIndex()
    except Exception as err:print (err)
    try: d['integR2'] = ui.integrationValue2.value()
    except Exception as err:print (err)
    try: d['integMin2'] = ui.acceptanceIntA2.value()
    except Exception as err:print (err)
    try: d['integMax2'] = ui.acceptanceIntB2.value()
    except Exception as err:print (err)
    try: d['Ch2relativeDeviation'] = np.array([ui.relativeDeviationValue.value(),0])
    except Exception as err:print(err)
    try: d['corrIlluPattern'] = ui.corrIlluPattern.isChecked()
    except Exception as err:print (err)
    try: d['enableMT'] = ui.EnableMT.isChecked()
    except Exception as err:print (err)

    # ui.plotLifetimeHist.setChecked(dl['plotLifetimeHist'])

    try: d['FRETn'] = ui.nFRETValue.value()
    except Exception as err:print (err)
    try: d['FRETAlpha'] = ui.FRETAlpha.value()
    except Exception as err:print (err)
    try: d['FRETDelta'] = ui.FRETDelta.value()
    except Exception as err:print (err)
    try: d['FRETGamma'] = ui.FRETGamma.value()
    except Exception as err:print (err)
    try: d['FRETBeta'] = ui.FRETBeta.value()
    except Exception as err:print (err)

    try: d['markEvent'] = ui.markEvent.isChecked()
    except Exception as err:print (err)
    try: d['markEventFrame'] = ui.markEventValue.value()
    except Exception as err:print (err)
    try: d['showFolds'] = ui.showFolds.isChecked()
    except Exception as err:print (err)
    try: d['showFoldsN'] = ui.showFoldsValue.value()
    except Exception as err:print (err)
    try: d['movingAvg'] = ui.movingAvg.isChecked()
    except Exception as err:print (err)
    try: d['movingAvgN'] = ui.movingAvgValue.value()
    except Exception as err:print (err)
    try: d['plotHMM'] = ui.plotHMM.isChecked()
    except Exception as err:print (err)
    try: d['HMMnStates'] = ui.HMMnStatesValue.value()
    except Exception as err:print (err)
    try: d['showPlots'] = ui.showPlots.isChecked()
    except Exception as err:print (err)
    try: d['savePlots'] = ui.savePlots.isChecked()
    except Exception as err:print (err)
    try: d['savePlotsFormatInd'] = ui.savePlotsFormat.currentIndex()
    except Exception as err:print (err)



def setSettings(ui, dl):
    ### QSettings not used yet!
    #https://python.hotexamples.com/examples/PyQt4.QtCore/QSettings/allKeys/python-qsettings-allkeys-method-examples.html
    # for key, value in dl.items():
    #     try:
    #         d[key] = (value)
    try: ui.saveFolder.setChecked(dl['saveOnDifferentFolder'])
    except: pass
    try: ui.saveFolderField.setText(dl['directoryAlt'])
    except: pass
    try: ui.recordingMode.setCurrentIndex(dl['recordingModeIndex'])
    except: pass
    try: ui.FPSValue.setValue(dl['FPS'])
    except: pass
    try: ui.ADfactorValue.setValue(dl['ADfactor'])
    except: pass
    try: ui.tempBinning.setChecked(dl['tempBin'])
    except: pass
    try: ui.tempBinningValue.setValue(dl['tempBinSize'])
    except: pass
    try: ui.spBinning.setChecked(dl['spBin'])
    except: pass
    try: ui.spBinningValue.setValue(dl['spBinSize'])
    except: pass
    try: ui.crop.setChecked(dl['crop'])
    except: pass
    try: ui.cropXA.setValue(dl['cropXA'])
    except: pass
    try: ui.cropXB.setValue(dl['cropXB'])
    except: pass
    try: ui.cropYA.setValue(dl['cropYA'])
    except: pass
    try: ui.cropYB.setValue(dl['cropYB'])
    except: pass
    try: ui.frameRange.setChecked(dl['LimitFrameRange'])
    except: pass
    try: ui.frameRangeA.setValue(dl['frameRangeA'])
    except: pass
    try: ui.frameRangeB.setValue(dl['frameRangeB'])
    except: pass
    try: ui.blurImg.setChecked(dl['blurImg'])
    except: pass
    try: ui.blurImgValue.setValue(dl['sigmaImg'])
    except: pass
    try: ui.removeBkgnd.setChecked(dl['removeBkgnd'])
    except: pass
    try: ui.bkgndFiltValue.setValue(dl['sigmaBkgnd'])
    except: pass
    try: ui.removeSpotsOfBkgnd.setChecked(dl['realBkgnd'])
    except: pass

    try: ui.avgForSpotDetectValue1.setValue(dl['FFS'])
    except: pass
    try: ui.seekingRvalue1.setValue(dl['seekR'])
    except: pass
    try: ui.ignorBlinking1.setChecked(dl['blinking'])
    except: pass
    try: ui.ignorBlinkingValue1.setValue(dl['blinkDuration'])
    except: pass
    try: ui.QEvalue1.setValue(dl['QE'])
    except: pass
    try: ui.integrationMode1.setCurrentIndex(dl['integrationModeIndex'])
    except: pass
    try: ui.integrationValue1.setValue(dl['integR'])
    except: pass
    try: 
        ui.acceptanceIntA1.setValue(dl['integMin'])
        ui.acceptanceIntB1.setValue(dl['integMax'])
        ui.AcceptingSpotIntCombo1.setCurrentIndex(1)
    except: pass

    try: ui.avgForSpotDetectValue2.setValue(dl['FFS2'])
    except: pass
    try: ui.seekingRvalue2.setValue(dl['seekR2'])
    except: pass
    try: ui.ignorBlinking2.setChecked(dl['blinking2'])
    except: pass
    try: ui.ignorBlinkingValue2.setValue(dl['blinkDuration2'])
    except: pass
    try: ui.QEvalue2.setValue(dl['QE2'])
    except: pass
    try: ui.integrationMode2.setCurrentIndex(dl['integrationModeIndex2'])
    except: pass
    try: ui.integrationValue2.setValue(dl['integR2'])
    except: pass
    try: 
        ui.acceptanceIntA2.setValue(dl['integMin2'])
        ui.acceptanceIntB2.setValue(dl['integMax2'])
        ui.AcceptingSpotIntCombo2.setCurrentIndex(1)
    except: pass
    try: ui.relativeDeviationValue.setValue(dl['Ch2relativeDeviation'])
    except: pass
    try: ui.corrIlluPattern.setChecked(dl['corrIlluPattern'])
    except: pass
    try: ui.EnableMT.setChecked(dl['enableMT'])
    except: pass

    # ui.plotLifetimeHist.setChecked(dl['plotLifetimeHist'])

    try: ui.nFRETValue.setValue(dl['FRETn'])
    except: pass
    try: ui.FRETAlpha.setValue(dl['FRETAlpha'])
    except: pass
    try: ui.FRETDelta.setValue(dl['FRETDelta'])
    except: pass
    try: ui.FRETGamma.setValue(dl['FRETGamma'])
    except: pass
    try: ui.FRETBeta.setValue(dl['FRETBeta'])
    except: pass

    try: ui.markEvent.setChecked(dl['markEvent'])
    except: pass
    try: ui.markEventValue.setValue(dl['markEventFrame'])
    except: pass
    try: ui.showFolds.setChecked(dl['showFolds'])
    except: pass
    try: ui.showFoldsValue.setValue(dl['showFoldsN'])
    except: pass
    try: ui.movingAvg.setChecked(dl['movingAvg'])
    except: pass
    try: ui.movingAvgValue.setValue(dl['movingAvgN'])
    except: pass
    try: ui.plotHMM.setChecked(dl['plotHMM'])
    except: pass
    try: ui.HMMnStatesValue.setValue(dl['HMMnStates'])
    except: pass
    try: ui.showPlots.setChecked(dl['showPlots'])
    except: pass
    try: ui.savePlots.setChecked(dl['savePlots'])
    except: pass
    try: ui.savePlotsFormat.setCurrentIndex(dl['savePlotsFormatInd'])
    except: pass
