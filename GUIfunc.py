from GUIdesign import *
from saverAndLoader import *
from plotter import *
from main import *
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
import os
import traceback
from PyQt5.QtWidgets import QFileDialog
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
from PyQt5.QtWidgets import QMessageBox
global d, dl
d = {}

# set app icon    
app_icon = QtGui.QIcon()
app_icon.addFile('gui/icons/16x16.png', QtCore.QSize(16,16))
app_icon.addFile('gui/icons/24x24.png', QtCore.QSize(24,24))
app_icon.addFile('gui/icons/32x32.png', QtCore.QSize(32,32))
app_icon.addFile('gui/icons/48x48.png', QtCore.QSize(48,48))
app_icon.addFile('gui/icons/256x256.png', QtCore.QSize(256,256))
app.setWindowIcon(app_icon)

##save and load settings

def status(text):
    status_bar = MainWindow.statusBar()
    status_bar.showMessage(text)


def actionSave_parametersClicked():
    print(d)
    d2={}
    getSettings(ui, d2)
    print(d2)

def actionAboutClicked():
    about = QMessageBox()
    about.setWindowTitle("About")
    about.setIcon(QMessageBox.Information)
    about.setText("____________________________________\n"
                  "               MT FRET analyzer\n"
                  "____________________________________\n"
                  "                 v 1.0  (2022-08) \n\n\n"
                  "               By: M.Sadegh Feiz \n"
                  "____________________________________\n"
                  "                        Enjoy!")
    x = about.exec_()
def sourceFolderBrowseClicked():
    filter = "TXT (*.txt);;PDF (*.pdf)"
    #fnames = QFileDialog.getOpenFileNames()
    iniDir = ''
    try: #to update initialSettings file
        iniFile = open("initialSettings.txt", "r")
        iniDir=iniFile.read()
        iniFile.close()
    except: pass
    print(iniDir)
    fnames = QFileDialog.getOpenFileNames(directory=iniDir)

    # file_name.setFileMode(QFileDialog.ExistingFiles)
    # names = file_name.getOpenFileNameAndFilter("Open files", "C\\", filter)

    for fname in fnames[0]:
        try:
            ui.sourceFolderField.setText(os.path.dirname(fname))
            ui.fileNamesField.insertPlainText(os.path.basename(fname + '\n'))
        except: traceback.print_exc()

def updateDir():
    from saverAndLoader import loader, setSettings
    from os.path import basename
    d['directorySource'] = ui.sourceFolderField.text() # update folder path if it is manually changed
    d['directoryAlt'] = ui.saveFolderField.text()
    d['fnamesExt'] = str.splitlines(ui.fileNamesField.toPlainText()) #update file list if it is manually changed
    if ui.saveFolder.isChecked():
        d['directorySaveParent'] = d['directoryAlt'] +'/'
    else:
        d['directorySaveParent'] = d['directorySource'] +'/'
    try:
        global dl #loaded d
        # load settings from (first) saved file to dictionary d
        d['fnameExt'] = d['fnamesExt'][0]
        d['fname'] = d['fnameExt'][0:d['fnameExt'].rfind('.')]
        d['directorySave'] = d['directorySaveParent'] + '/' + d['fname'] + '/'        
        dl = loader(d['directorySave'], d['fname'])
        # set the parameters on GUI
        setSettings(ui, dl)
        print('Settings loaded')
    except: pass

def saveFolderChanged():
    ui.saveFolderField.setEnabled(ui.saveFolder.isChecked())
    ui.saveFolderBrowse.setEnabled(ui.saveFolder.isChecked())
    d['saveOnDifferentFolder'] = ui.saveFolder.isChecked()
    updateDir()

def saveFolderFieldChanged():
    updateDir()

def saveFolderBrowseClicked():
    fname = QFileDialog.getExistingDirectory()
    try:
        ui.saveFolderField.setText(fname)
    except: traceback.print_exc()
    updateDir()

def recordingModeChanged():
    if ui.recordingMode.currentIndex() == 0:
        d['recordingMode'] = 'singleChannel'
    if ui.recordingMode.currentIndex() == 1:
        d['recordingMode'] = 'singleChannelDoubleDye'
    if ui.recordingMode.currentIndex() == 2:
        d['recordingMode'] = 'alex'
    if ui.recordingMode.currentIndex() == 3:
        d['recordingMode'] = 'firstFrameDifferent'

def FPSChanged():
    d['FPS'] = ui.FPSValue.value()

def ADfactorChanged():
    d['ADfactor'] = ui.ADfactorValue.value()
    d['ADfactor2'] = ui.ADfactorValue.value()
def tempBinningChanged():
    ui.tempBinningValue.setEnabled(ui.tempBinning.isChecked())
    ui.tempBinningLabel.setEnabled(ui.tempBinning.isChecked())
    if ui.tempBinning.isChecked():
        d['tempBinSize'] = ui.tempBinningValue.value()
    else:
        d['tempBinSize'] = 1

def spBinningChanged():
    ui.spBinningValue.setEnabled(ui.spBinning.isChecked())
    ui.spBinningLabel.setEnabled(ui.spBinning.isChecked())
    if ui.spBinning.isChecked():
        d['spBinSize'] = ui.spBinningValue.value()
    else:
        d['spBinSize'] = 1

def cropChanged():
    ui.cropXA.setEnabled(ui.crop.isChecked())
    ui.cropXB.setEnabled(ui.crop.isChecked())
    ui.cropYA.setEnabled(ui.crop.isChecked())
    ui.cropYB.setEnabled(ui.crop.isChecked())
    ui.cropLabel.setEnabled(ui.crop.isChecked())
    ui.cropSelect.setEnabled(ui.crop.isChecked())
    d['cropXA'] = ui.cropXA.value()
    d['cropXB'] = ui.cropXB.value()
    d['cropYA'] = ui.cropYA.value()
    d['cropYB'] = ui.cropYB.value()

def frameRangeChanged():
    ui.frameRangeA.setEnabled(ui.frameRange.isChecked())
    ui.frameRangeB.setEnabled(ui.frameRange.isChecked())
    d['frameRangeA'] = ui.frameRangeA.value()
    d['frameRangeB'] = ui.frameRangeB.value()

def blurImgChanged():
    ui.blurImgValue.setEnabled(ui.blurImg.isChecked())
    ui.blurImgLabel.setEnabled(ui.blurImg.isChecked())
    if ui.blurImg.isChecked():
        d['sigmaImg']  = ui.blurImgValue.value()
    else:
        d['sigmaImg']  = 0

def removeBkgndChanged():
    ui.bkgndFiltValue.setEnabled(ui.removeBkgnd.isChecked())
    ui.removeBkgndLabel.setEnabled(ui.removeBkgnd.isChecked())
    ui.removeSpotsOfBkgnd.setEnabled(ui.removeBkgnd.isChecked())
    if ui.removeBkgnd.isChecked():
        d['sigmaBkgnd'] = ui.bkgndFiltValue.value()
        d['realBkgnd'] = ui.removeSpotsOfBkgnd.isChecked()
    else:
        d['sigmaBkgnd'] = 0
        d['realBkgnd'] = 0

def avgForSpotDetectValue1changed():
    d['FFS'] = ui.avgForSpotDetectValue1.value()

def seekingRvalue1changed():
    d['seekR'] = ui.seekingRvalue1.value()

def ignorBlinking1Changed():
    ui.ignorBlinkingLabel1.setEnabled(ui.ignorBlinking1.isChecked())
    ui.ignorBlinkingValue1.setEnabled(ui.ignorBlinking1.isChecked())
    if ui.ignorBlinking1.isChecked():
        d['blinkDuration'] = ui.ignorBlinkingValue1.value()
    else:
        d['blinkDuration'] = 0

def QEvalue1Changed():
    d['QE'] = ui.QEvalue1.value()

def integrationMode1Changed():
    if ui.integrationMode1.currentIndex() == 0:
        ui.integrationValue1.setEnabled(1)
        ui.integrationLabel1.setEnabled(1)
        d['integralSpotShape'] = 'circle'
        d['integR'] = ui.integrationValue1.value()
    if ui.integrationMode1.currentIndex() == 1:
        ui.integrationValue1.setEnabled(0)
        ui.integrationLabel1.setEnabled(0)
        d['integralSpotShape'] = 'average'
    if ui.integrationMode1.currentIndex() == 2:
        ui.integrationValue1.setEnabled(0)
        ui.integrationLabel1.setEnabled(0)
        d['integralSpotShape'] = 'fit'

def AcceptingSpotIntCombo1Changed():
    if ui.AcceptingSpotIntCombo1.currentIndex() == 0:
        ui.acceptanceIntA1.setEnabled(0)
        ui.acceptanceIntB1.setEnabled(0)
        ui.label_3.setEnabled(0)
    else:
        ui.acceptanceIntA1.setEnabled(1)
        ui.acceptanceIntB1.setEnabled(1)
        ui.label_3.setEnabled(1)
    d['AcceptingSpotIntCombo'] = ui.AcceptingSpotIntCombo1.currentIndex()

def acceptanceInt1Changed():
    d['integMin'] = ui.acceptanceIntA1.value()
    d['integMax'] = ui.acceptanceIntB1.value()

def avgForSpotDetectValue2changed():
    d['FFS2'] = ui.avgForSpotDetectValue2.value()

def seekingRvalue2changed():
    d['seekR2'] = ui.seekingRvalue2.value()

def ignorBlinking2Changed():
    ui.ignorBlinkingLabel2.setEnabled(ui.ignorBlinking2.isChecked())
    ui.ignorBlinkingValue2.setEnabled(ui.ignorBlinking2.isChecked())
    if ui.ignorBlinking2.isChecked():
        d['blinkDuration2'] = ui.ignorBlinkingValue2.value()
    else:
        d['blinkDuration2'] = 0

def QEvalue2Changed():
    d['QE2'] = ui.QEvalue2.value()

def integrationMode2Changed():
    if ui.integrationMode2.currentIndex() == 0:
        ui.integrationValue2.setEnabled(1)
        ui.integrationLabel2.setEnabled(1)
        d['integralSpotShape2'] = 'circle'
        d['integR2'] = ui.integrationValue2.value()
    if ui.integrationMode2.currentIndex() == 1:
        ui.integrationValue2.setEnabled(0)
        ui.integrationLabel2.setEnabled(0)
        d['integralSpotShape2'] = 'average'
    if ui.integrationMode2.currentIndex() == 2:
        ui.integrationValue2.setEnabled(0)
        ui.integrationLabel2.setEnabled(0)
        d['integralSpotShape2'] = 'fit'

def AcceptingSpotIntCombo2Changed():
    if ui.AcceptingSpotIntCombo2.currentIndex() == 0:
        ui.acceptanceIntA2.setEnabled(0)
        ui.acceptanceIntB2.setEnabled(0)
        ui.label_2.setEnabled(0)
    else:
        ui.acceptanceIntA2.setEnabled(1)
        ui.acceptanceIntB2.setEnabled(1)
        ui.label_2.setEnabled(1)
    d['AcceptingSpotIntCombo2'] = ui.AcceptingSpotIntCombo2.currentIndex()


def acceptanceInt2Changed():
    d['integMin2'] = ui.acceptanceIntA2.value()
    d['integMax2'] = ui.acceptanceIntB2.value()

def relativeDeviationValueChanged():
    d['alexSpotDist'] = [ui.relativeDeviationValue.value(), 0] # defined as [dy, dx]

def corrIlluPatternChanged():
    d['correctExPatterns'] = ui.corrIlluPattern.isChecked()

def EnableMTChanged():
    pass

def frameRangePlotChanged():
    ui.frameRangePlotA.setEnabled(ui.frameRangePlot.isChecked())
    ui.frameRangePlotB.setEnabled(ui.frameRangePlot.isChecked())
    ui.frameRangePlotLabel.setEnabled(ui.frameRangePlot.isChecked())
    d['frameRangePlot'] = ui.frameRangePlot.isChecked()
    d['frameRangePlotA'] = ui.frameRangePlotA.value()
    d['frameRangePlotB'] = ui.frameRangePlotB.value()


def markEventChanged():
    ui.markEventValue.setEnabled(ui.markEvent.isChecked())
    ui.markEventLabel.setEnabled(ui.markEvent.isChecked())
    d['markEventFrame'] = ui.markEventValue.value()


def showFoldsChanged():
    d['showFolds'] = ui.showFolds.isChecked()
    ui.showFoldsLabel.setEnabled(ui.showFolds.isChecked())
    ui.showFoldsValue.setEnabled(ui.showFolds.isChecked())
    d['foldN'] = ui.showFoldsValue.value()


def movingAvgChanged():
    ui.movingAvgLabel.setEnabled(ui.movingAvg.isChecked())
    ui.movingAvgValue.setEnabled(ui.movingAvg.isChecked())
    if ui.movingAvg.isChecked():
        d['movingAvgN'] = ui.movingAvgValue.value()
    else:
        d['movingAvgN'] = 1

def plotFRETChanged():
    if ui.plotSE.isChecked() or ui.plotFRETtraces.isChecked():
        ui.FRET.setEnabled(1)
        ui.FRETManuallCoefs.setEnabled(1)#BUG!
    else:
        ui.FRET.setEnabled(0)
    FRETcalcChanged()

def plotHMMChanged():
    ui.HMMnStatesValue.setEnabled(ui.plotHMM.isChecked())
    ui.HMMnStatesLabel.setEnabled(ui.plotHMM.isChecked())
    d['plotHMM'] = ui.plotHMM.isChecked()
    d['HMMnStates'] = ui.HMMnStatesValue.value()

def FRETcalcChanged():
    if ui.FRETFindCoefs.isChecked():
        ui.nFRET.setEnabled(1)
        ui.nFRETValue.setEnabled(1)
        ui.FRETCoefs.setEnabled(0)
        ui.FRETAlpha.setEnabled(0)
        ui.FRETDelta.setEnabled(0)
        ui.FRETGamma.setEnabled(0)
        ui.FRETBeta.setEnabled(0)
        d['numberOfPopulations'] = ui.nFRETValue.value()
    else:
        ui.nFRET.setEnabled(0)
        ui.nFRETValue.setEnabled(0)
        ui.FRETCoefs.setEnabled(1)
        ui.FRETAlpha.setEnabled(1)
        ui.FRETDelta.setEnabled(1)
        ui.FRETGamma.setEnabled(1)
        ui.FRETBeta.setEnabled(1)
        d['FRETAlpha'] = ui.FRETAlpha.value()
        d['FRETDelta'] = ui.FRETDelta.value()
        d['FRETGamma'] = ui.FRETGamma.value()
        d['FRETBeta'] = ui.FRETBeta.value()

def plotConfig():
    d['showPlots'] = ui.showPlots.isChecked()
    d['savePlots'] = ui.savePlots.isChecked()
    if ui.savePlotsFormat.currentIndex() == 0:
        d['savePlotsFormat'] = '.png'
    if ui.savePlotsFormat.currentIndex() == 1:
        d['savePlotsFormat'] = '.eps'
    if ui.savePlotsFormat.currentIndex() == 2:
        d['savePlotsFormat'] = '.pdf'
    if ui.savePlots.isChecked():
        ui.savePlotsFormat.setEnabled(1)
    else:
        ui.savePlotsFormat.setEnabled(0)

def rePlotClicked():
    import time
    print('Plotting...', end='')
    time.sleep(0.05)
    try:
    # load data from .xls file:
        try:
            d['directorySave'] = d['directorySaveParent'] + '/' + d['fname'] + '/'
            dl = loader(d['directorySave'], d['fname'] ) # to replace new generated data with old loaded one
            dl['fname'] = d['fnameExt'][0:d['fnameExt'].rfind('.')]# to correct if the file name is changed manually
            dl['directorySave'] = d['directorySaveParent'] + '/' + dl['fname'] + '/'
        except: traceback.print_exc()
        Path(dl['directorySave']).mkdir(parents=True, exist_ok=True)
        if ui.plotStatistics.isChecked():
            plot1(dl)
            plot2(dl)
            plot3(dl)
            plot4(dl)
            plot6(dl)
        if ui.plotIntAll.isChecked():
            plot5(dl)
        if ui.plotIntTraces.isChecked():
            plot7(dl)
        if ui.plotFRETtraces.isChecked():
            plot7FRET(dl)
        if ui.plotSE.isChecked():
            FRETcalc(dl)
        print('done!')
    except: traceback.print_exc()

def startClicked():
    import os
    import time
    print('Start', end='')
    time.sleep(0.05)
    os.system('cls' if os.name == 'nt' else 'clear') #clear terminal
    ### correct for directory typo
    if ui.sourceFolderField.text()[-1:] == '/':
        ui.sourceFolderField.setText(ui.sourceFolderField.text()[:-1])
    if ui.saveFolderField.text()[-1:] == '/':
        ui.saveFolderField.setText(ui.saveFolderField.text()[:-1])
    # updateDir()
    getSettings(ui, d)
    
    
    try: #to update initialSettings file
        iniFile = open("initialSettings.txt", "w")
        iniFile.write(d['directorySaveParent'])
        iniFile.close()
    except: pass
    # ui.centralwidget.setEnabled(0)
    for d['fnameExt'] in d['fnamesExt']:
        analyzeMaster(d)
        rePlotClicked()
    # ui.centralwidget.setEnabled(1)

def stopClicked():
    print('stopping...')
    import sys
    sys.exit()

def runGUI():
    import time
    print('Start', end='')
    time.sleep(0.05)
    ### set Ch.1 tab selected:
    ui.channels.setCurrentIndex(0)
    ### To have an idea of first values:
    updateDir()
    recordingModeChanged()
    FPSChanged()
    tempBinningChanged()
    spBinningChanged()
    cropChanged()
    frameRangeChanged()
    blurImgChanged()
    removeBkgndChanged()
    avgForSpotDetectValue1changed()
    seekingRvalue1changed()
    ignorBlinking1Changed()
    QEvalue1Changed()
    integrationMode1Changed()
    AcceptingSpotIntCombo1Changed()
    acceptanceInt1Changed()
    avgForSpotDetectValue2changed()
    seekingRvalue2changed()
    ignorBlinking2Changed()
    QEvalue2Changed()
    integrationMode2Changed()
    AcceptingSpotIntCombo2Changed()
    acceptanceInt2Changed()
    relativeDeviationValueChanged()
    corrIlluPatternChanged()
    EnableMTChanged()
    markEventChanged()
    showFoldsChanged()
    frameRangePlotChanged()
    movingAvgChanged()
    plotFRETChanged()
    plotHMMChanged()
    FRETcalcChanged()
    plotConfig()
    print('.', end='')
    ### interactivity:
    ### Menu
    ui.actionSave_parameters.triggered.connect(actionSave_parametersClicked)
    ui.actionAbout.triggered.connect(actionAboutClicked)
    ### first column
    ui.sourceFolderBrowse.clicked.connect(sourceFolderBrowseClicked)
    ui.fileNamesField.textChanged.connect(updateDir)
    ui.sourceFolderField.textChanged.connect(updateDir)
    ui.saveFolder.stateChanged.connect(saveFolderChanged)
    ui.saveFolderBrowse.clicked.connect(saveFolderBrowseClicked)
    ui.saveFolderField.textChanged.connect(saveFolderFieldChanged)
    ui.recordingMode.currentIndexChanged.connect(recordingModeChanged)
    ui.FPSValue.valueChanged.connect(FPSChanged)
    ui.ADfactorValue.valueChanged.connect(ADfactorChanged)
    ui.tempBinning.stateChanged.connect(tempBinningChanged)
    ui.tempBinningValue.valueChanged.connect(tempBinningChanged)
    ui.spBinning.stateChanged.connect(spBinningChanged)
    ui.spBinningValue.valueChanged.connect(spBinningChanged)
    ui.crop.stateChanged.connect(cropChanged)
    ui.cropXA.valueChanged.connect(cropChanged)
    ui.cropXB.valueChanged.connect(cropChanged)
    ui.cropYA.valueChanged.connect(cropChanged)
    ui.cropYB.valueChanged.connect(cropChanged)
    ui.frameRange.stateChanged.connect(frameRangeChanged)
    ui.frameRangeA.valueChanged.connect(frameRangeChanged)
    ui.frameRangeB.valueChanged.connect(frameRangeChanged)
    ui.blurImg.stateChanged.connect(blurImgChanged)
    ui.blurImgValue.valueChanged.connect(blurImgChanged)
    ui.removeBkgnd.stateChanged.connect(removeBkgndChanged)
    ui.bkgndFiltValue.valueChanged.connect(removeBkgndChanged)
    ui.removeSpotsOfBkgnd.stateChanged.connect(removeBkgndChanged)
    ### Ch.1:
    ui.avgForSpotDetectValue1.valueChanged.connect(avgForSpotDetectValue1changed)
    ui.seekingRvalue1.valueChanged.connect(seekingRvalue1changed)
    ui.ignorBlinking1.stateChanged.connect(ignorBlinking1Changed)
    ui.ignorBlinkingValue1.valueChanged.connect(ignorBlinking1Changed)
    ui.QEvalue1.valueChanged.connect(QEvalue1Changed)
    ui.integrationMode1.currentIndexChanged.connect(integrationMode1Changed)
    ui.integrationValue1.valueChanged.connect(integrationMode1Changed)
    ui.AcceptingSpotIntCombo1.currentIndexChanged.connect(AcceptingSpotIntCombo1Changed)
    ui.acceptanceIntA1.valueChanged.connect(acceptanceInt1Changed)
    ui.acceptanceIntB1.valueChanged.connect(acceptanceInt1Changed)
    ### Ch.2:
    ui.avgForSpotDetectValue2.valueChanged.connect(avgForSpotDetectValue2changed)
    ui.seekingRvalue2.valueChanged.connect(seekingRvalue2changed)
    ui.ignorBlinking2.stateChanged.connect(ignorBlinking2Changed)
    ui.ignorBlinkingValue2.valueChanged.connect(ignorBlinking2Changed)
    ui.QEvalue2.valueChanged.connect(QEvalue2Changed)
    ui.integrationMode2.currentIndexChanged.connect(integrationMode2Changed)
    ui.integrationValue2.valueChanged.connect(integrationMode2Changed)
    ui.AcceptingSpotIntCombo2.currentIndexChanged.connect(AcceptingSpotIntCombo2Changed)
    ui.acceptanceIntA2.valueChanged.connect(acceptanceInt2Changed)
    ui.acceptanceIntB2.valueChanged.connect(acceptanceInt2Changed)
    ui.relativeDeviationValue.valueChanged.connect(relativeDeviationValueChanged)
    ui.corrIlluPattern.stateChanged.connect(corrIlluPatternChanged)
    ### MT:
    ui.EnableMT.stateChanged.connect(EnableMTChanged)
    ###plots
    ui.plotSE.stateChanged.connect(plotFRETChanged)
    ui.plotFRETtraces.stateChanged.connect(plotFRETChanged)
    ui.plotHMM.stateChanged.connect(plotHMMChanged)
    ui.HMMnStatesValue.valueChanged.connect(plotHMMChanged)
    ui.FRETFindCoefs.toggled.connect(FRETcalcChanged)
    ui.frameRangePlot.stateChanged.connect(frameRangePlotChanged)
    ui.frameRangePlotA.valueChanged.connect(frameRangePlotChanged)
    ui.frameRangePlotB.valueChanged.connect(frameRangePlotChanged)
    ui.markEvent.stateChanged.connect(markEventChanged)
    ui.showFolds.stateChanged.connect(showFoldsChanged)
    ui.movingAvg.stateChanged.connect(movingAvgChanged)
    ui.movingAvgValue.valueChanged.connect(movingAvgChanged)
    ui.showPlots.stateChanged.connect(plotConfig)
    ui.savePlots.stateChanged.connect(plotConfig)
    ui.savePlotsFormat.currentIndexChanged.connect(plotConfig)
    ### replot
    ui.rePlot.clicked.connect(rePlotClicked)
    ui.start.clicked.connect(startClicked)
    ui.stop.clicked.connect(stopClicked)
    status('Ready!')
    MainWindow.show()
    sys.exit(app.exec_())
