import numpy as np
import matplotlib.pyplot as plt
global x1
global y1
xHist = np.arange(-10,10,0.1)
yHist = 10000-xHist**3

fig = plt.figure()
ax = fig.add_subplot(111)

# class clicking():
#     def onclick(self, event):
#         ax = self.figure.gca()
#         ax.cla()
#         binwidth = 100
#         n, bins, patches = plt.hist(yHist, bins=range(1, int(yHist.max() + binwidth), binwidth), align='mid', facecolor='grey' )    
#         ix, iy = event.xdata, event.ydata
#         x = round(event.xdata, 5)
#         if event.button == 1 and len(ax.lines) <= 2: ## 3, 2, 1 for right, middle and left click
#             ax.axvline(x, color='g', alpha=0.7)
#             self.canvas.draw()
#         print(f'x = {ix}')

#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
#     plt.show()




class myclass():
    def plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111) #self.
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel(r'z position ($\mu$m)')
        self.start_value = None # this is to reset these values for the current dataset
        self.end_value = None
        # self.ax.hold(True) this function is deprecated and is by default already set to true.to clear axes, use the ax.cla ()
        ### draw the plot in the window canvas ###
        self.canvas.draw()
        ### activate/connect mouse clicks to the plot --> such that mouse clicks can define start and end values of selection
        self.canvas.mpl_connect('button_press_event', myclass.onclick)
            
    def onclick(self, event):
        x = round(event.xdata, 5)
        print('Clicked at x:', x)
        print('Clicked:', event.button)

        ax = self.figure.gca()
        #When clicking start and end of trace
        if event.button == 3 and len(ax.lines) <= 2: ## 3, 2, 1 for right, middle and left click
            ax.axvline(x, color='g', alpha=0.7)
            self.canvas.draw()

            if len(ax.lines) == 3:
                ln1 = ax.lines[1].get_xdata()
                ln2 = ax.lines[2].get_xdata()
                if ln1[0] > ln2[0]:
                    self.xline = ln1
                    self.save_end_value()
                    self.xline = ln2
                    self.save_start_value()
                else:
                    self.xline = ln2
                    self.save_end_value()
                    self.xline = ln1
                    self.save_start_value()
        elif event.button == 3 and len(ax.lines) >= 3:
            self.statusbar.showMessage('You already drew two lines! Please discard dataset to redraw lines.')
            print('You already drew two lines! Please discard dataset to redraw lines.')
            # self.infotext.te

        #When removing a line
        if event.button == 1 and event.dblclick and len(ax.lines) > 1: ## 3, 2, 1 for right, middle and left click
            ax = self.figure.gca()
            ax.lines[-1].remove()
            self.canvas.draw()

        #When clicking the start of loading
        if event.button == 2 and self.loadCB.isChecked(): ## 3, 2, 1 for right, middle and left click
            ax = self.figure.gca()
            ax.axvline(x, color='r', alpha=0.7)
            self.load_start = x
            print('Loading start registered!')
            self.canvas.draw()
    def final(self):
        self.plot()
pl=myclass()
pl.final()
plt.show()
pl.onclick()
