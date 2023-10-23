from saverAndLoader import *
from plotter import *
directory = 'C:\\Feiz\\20220426\\bs1s20\\'
# directory = 'C:\\Feiz\\+testFRET\\'
fileName = 'b.tif'
d = loader(directory, fileName)
########################################################
# d = plot1(d) # life-time histogram
# d = plot2(d) # photon count histogram
# d = plot3(d) # Average No. of photons / frame / dye (David)
### d = plot4(d) # fitting a surface to the spots mean intensity to see homeogenity of FOV (David)
# d = plot6(d) # spots vs time (Ibrahim)
# d = plot7(d) # intensity track one by one
d = plot7FRET(d) # FRET track one by one
# d = plot5(d) # intensity track plot
# d = FRETcalc(d)
# HMMplot(d)
winsound.Beep(frequency=1500, duration=500)