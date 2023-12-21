import numpy as np
import matplotlib.pyplot as plt
import skimage

imgX = 2048
imgY = 2048
backgroundLevel = 20#150
signalLevelD = 1
signalLevelA = 1
distance = 10
nFrame = 20
path=''
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication
app = QApplication([])
def ALEX_FRET_Simulator():
    #gui to get the parameters
    dialog = QDialog()
    dialog.setWindowTitle("Image simulator")
    dialog.resize(400, 300)
    dialog.setModal(True)
    # dialog.setWindowIcon(app_icon)
    layout = QVBoxLayout()
    dialog.setLayout(layout)
    #img size
    layout1 = QHBoxLayout()
    layout.addLayout(layout1)
    label = QLabel("Image size:")
    layout1.addWidget(label)
    label = QLabel("      x:")
    layout1.addWidget(label)
    imgX = QSpinBox()
    imgX.setMinimum(1)
    imgX.setMaximum(10000)
    imgX.setValue(2048)
    layout1.addWidget(imgX)
    label = QLabel("      y:")
    layout1.addWidget(label)
    imgY = QSpinBox()
    imgY.setMinimum(1)
    imgY.setMaximum(10000)
    imgY.setValue(2048)
    layout1.addWidget(imgY)
    
    #background level
    layout2 = QHBoxLayout()
    layout.addLayout(layout2)
    label = QLabel("Background level:")
    layout2.addWidget(label)
    backgroundLevel = QSpinBox()
    backgroundLevel.setMinimum(0)
    backgroundLevel.setMaximum(65000)
    backgroundLevel.setValue(100)
    layout2.addWidget(backgroundLevel)
    #signal level
    layout3 = QHBoxLayout()
    layout.addLayout(layout3)
    label = QLabel("Signal level,    ")
    layout3.addWidget(label)
    label = QLabel("Donor:")
    layout3.addWidget(label)
    signalLevelD = QSpinBox()
    signalLevelD.setMinimum(0)
    signalLevelD.setMaximum(1000)
    signalLevelD.setValue(10)
    layout3.addWidget(signalLevelD)
    label = QLabel("Acceptor:")
    layout3.addWidget(label)
    signalLevelA = QSpinBox()
    signalLevelA.setMinimum(0)
    signalLevelA.setMaximum(1000)
    signalLevelA.setValue(10)
    layout3.addWidget(signalLevelA)
    #distance
    layout4 = QHBoxLayout()
    layout.addLayout(layout4)
    label = QLabel("Distance between A and D spots:")
    layout4.addWidget(label)
    distance = QSpinBox()
    distance.setMinimum(0)
    distance.setMaximum(100)
    distance.setValue(10)
    layout4.addWidget(distance)
    label = QLabel("pixels")
    layout4.addWidget(label)
    #nFrame
    layout5 = QHBoxLayout()
    layout.addLayout(layout5)
    label = QLabel("Number of frames per channel:")
    layout5.addWidget(label)
    nFrame = QSpinBox()
    nFrame.setMinimum(1)
    nFrame.setMaximum(1000)
    nFrame.setValue(50)
    layout5.addWidget(nFrame)
    #path
    layout6 = QHBoxLayout()
    layout.addLayout(layout6)
    label = QLabel("Save path:")
    layout6.addWidget(label)
    
    # add a "brwose" botton
    browse_button = QPushButton("Browse")
    layout6.addWidget(browse_button)

    def browse():
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(filter="TIFF files (*.tif)", caption="Save as", directory="../simulated_image.tif")
        path.setText(file_path)

    browse_button.clicked.connect(browse)
    layout7 = QHBoxLayout()
    layout.addLayout(layout7)
    path = QLineEdit()
    layout7.addWidget(path)
    #buttons
    layout8 = QHBoxLayout()
    layout.addLayout(layout8)
    button = QPushButton("OK")
    button.clicked.connect(dialog.accept)
    layout8.addWidget(button)
    button = QPushButton("Cancel")
    button.clicked.connect(dialog.reject)
    layout8.addWidget(button)
    dialog.exec_()
    #get the parameters 
    imgX = imgX.value()
    imgY = imgY.value()
    backgroundLevel = backgroundLevel.value()
    signalLevelD = signalLevelD.value()
    signalLevelA = signalLevelA.value()
    distance = distance.value()
    nFrame = nFrame.value()
    #if "ok" botton in GUI is pressed, print something
    if dialog.result() == QDialog.Accepted:
        channel1 = np.zeros([imgY, imgX])
        channel2 = np.zeros([imgY, imgX])
        method = 2

        if method == 1:
            r = 4
            for sn in range(1500):
                i = 20 + int(np.random.random(1) * (imgY-40))
                j = 20 + int(np.random.random(1) * (imgX-40))
                PD = np.random.random(1)# D-only
                PHF = np.random.random(1)# high fret
                PDA = np.random.random(1)
                for ii in range(-5,5):
                    for jj in range(-5,5):
                        if PD > 0.1: # not D-only
                            channel2[i + ii + distance, j + jj] = channel2[i + ii + distance, j + jj] + signalLevelA * max(r - np.sqrt(ii**2 + jj**2), 0)
                            if PDA > 0.1:# it is D-A pair, not A-only or D-only
                                if PHF < 0.5: # low fret
                                    channel1[i + ii, j + jj] = channel1[i + ii, j + jj] + signalLevelD * max(r - np.sqrt(ii**2 + jj**2), 0)
                                else:
                                    channel1[i + ii + distance, j + jj] = channel1[i + ii + distance, j + jj] + signalLevelA * max(r - np.sqrt(ii**2 + jj**2), 0)
                        else:
                            channel1[i + ii, j + jj] = channel1[i + ii, j + jj] + signalLevelD * max(r - np.sqrt(ii**2 + jj**2), 0)




        if method == 2:
            spotindex2=np.array([[-3, -3 ,-3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4],
                        [-4, -3, -2, -1,  0,  1,  2,  3, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5, -4, -3, -2, -1,  0,  1,  2,  3, -4, -3, -2, -1,  0,  1,  2,  3,  0],
                        [11.85349462, 11.55376344, 14.19892473, 18.62768817, 19.10483871, 16.03360215, 13.22177419,  8.95967742,  9.89112903, 12.49327957, 16.86693548, 20.48924731, 26.47043011, 35.12634409, 38.32526882, 36.06854839, 26.17607527, 16.91935484, 13.78091398, 17.03629032, 23.69354839, 30.16935484, 42.2016129,  53.15860215,
                        62.67607527, 57.42876344, 40.51747312, 25.9327957,  12.14247312, 10.27150538, 15.96639785, 21.64516129, 26.21639785, 35.14784946, 50.26478495, 63.94758065, 72.05241935, 66.01612903, 47.98790323, 27.50268817, 14.1438172,  14.82930108, 17.94892473, 24.03897849, 30.84811828, 42.97849462, 55.12903226, 61.97715054,
                        54.27150538, 40.37903226, 24.0483871,  11.84274194, 13.00268817, 15.68413978, 20.31586022, 26.71370968, 34.06182796, 37.36155914, 34.93817204, 25.96236559, 14.26344086, 10.50403226, 11.97043011, 13.57258065, 16.14112903, 18.16397849,
                        17.82795699, 13.88575269, 10.18413978, 10.36693548]])

            spotindex1=np.array([[-3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3, 3,  3,  3],
                        [-3, -2, -1,  0,  1,  2,  3, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2, 3,  4,  5, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0, 1,  2,  3],
                        [ 6.74078212,  8.86592179,  8.79106145,  8.83240223,  7.97877095,  8.93519553, 5.56759777,  8.42458101,  8.84916201, 10.05027933, 12.16648045, 15.06145251, 16.06592179, 17.81005587, 18.72178771, 18.17988827, 12.01564246,  8.1452514, 12.46815642,  9.73184358, 11.62458101, 15.03575419, 20.04804469, 25.69162011,
                        27.06927374, 32.66256983, 31.83687151, 27.60111732, 21.96424581, 14.31843575, 8.92513966, 12.86145251, 11.43798883, 13.01340782, 18.03128492, 23.25139665, 25.57206704, 35.8603352,  37.88044693, 38.86480447, 32.33519553, 23.91061453, 16.48938547,  8.20335196, 13.44692737,  9.9575419,  12.50391061, 14.13743017,
                        19.72067039, 24.99329609, 28.88156425, 31.17653631, 30.54413408, 27.99329609, 19.81005587, 14.74972067, 10.58324022,  8.26703911, 10.3396648,  10.2301676, 13.09608939, 14.76089385, 17.30837989, 19.09497207, 17.82681564, 16.4849162, 10.95195531,  8.44804469,  8.61117318,  7.88268156,  7.96871508,  7.01564246,
                        9.18659218,  8.50167598,  6.03687151]])
            dist = -distance
            # FRET pair:
            for E in [0, 0.25, 0.5, 0.75, 1]:
                for sn in range(200):
                    i = 20 + int(np.random.random(1) * (imgY-40))
                    j = 20 + int(np.random.random(1) * (imgX-40))
                    for jj, ii, kk in zip(*spotindex1):
                        ii = int(ii)
                        jj = int(jj)
                        channel1[i +ii , j + jj] = channel1[i + ii, j + jj] + signalLevelD * (1 - E) * kk
                        channel1[i +ii + dist, j + jj] = channel1[i + ii + dist, j + jj] + signalLevelD * E * kk
                    for jj, ii, kk in zip(*spotindex2):
                        ii = int(ii)
                        jj = int(jj)
                        channel2[i +ii + dist, j + jj] = channel2[i + ii + dist, j + jj] + signalLevelA * kk
            # D-only spots
            for sn in range(200):
                i = 20 + int(np.random.random(1) * (imgY-40))
                j = 20 + int(np.random.random(1) * (imgX-40))
                for jj, ii, kk in zip(*spotindex1):
                    ii = int(ii)
                    jj = int(jj)
                    channel1[i +ii , j + jj] = channel1[i + ii, j + jj] + signalLevelD * kk

            # A-only spots
            for sn in range(200):
                i = 20 + int(np.random.random(1) * (imgY-40))
                j = 20 + int(np.random.random(1) * (imgX-40))
                for jj, ii, kk in zip(*spotindex2):
                    ii = int(ii)
                    jj = int(jj)
                    channel2[i +ii + dist, j + jj] = channel2[i + ii + dist, j + jj] + signalLevelA * kk

        plt.imshow(channel1, cmap='gray', vmin = 0, vmax=np.max(channel1))
        plt.axis('off')
        plt.show()
        img = [channel1+ backgroundLevel * (np.random.random([imgY, imgX])),channel2 + backgroundLevel * (np.random.random([imgY, imgX]))]
        for i in range(nFrame):
            img.append(channel1+ backgroundLevel * (np.random.random([imgY, imgX])))
            img.append(channel2+ backgroundLevel * (np.random.random([imgY, imgX])))
        img = np.asarray(img)
        img = img.astype('int16')
        # Save the simulated image
        skimage.io.imsave(path.text(), img, photometric='minisblack')

ALEX_FRET_Simulator()