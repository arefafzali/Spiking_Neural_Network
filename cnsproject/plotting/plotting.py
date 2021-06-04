"""
Module for visualization and plotting.

TODO.

Implement this module in any way you are comfortable with. You are free to use\
any visualization library you like. Providing live plotting and animations is\
also a bonus. The visualizations you will definitely need are as follows:

1. F-I curve.
2. Voltage/current dynamic through time.
3. Raster plot of spikes in a neural population.
4. Convolutional weight demonstration.
5. Weight change through time.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch


class plotting():

    def reset(self):
        self.figure = plt.figure(figsize=(10,5))
        self.colors = plt.rcParams["axes.prop_cycle"]()


    def show(self):
        self.figure.legend()
        plt.show()
        return


    def plot_ut_it_init(self, time) -> None:
        self.reset()
        self.ax1 = self.figure.add_subplot(121)
        self.ax1.set_ylabel('I(t)')
        self.ax1.set_xlabel('time')
        self.ax1.set_title("Current per second")
        self.ax1.grid(True)
        self.ax1.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax1.set_xticklabels(np.arange(time+1, dtype=int))
        
        self.ax2 = self.figure.add_subplot(122)
        self.ax2.set_ylabel('U(t)')
        self.ax2.set_xlabel('time')
        self.ax2.set_title("Potential per second")
        self.ax2.grid(True)
        self.ax2.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax2.set_xticklabels(np.arange(time+1, dtype=int))
        return


    def plot_ut_it_update(self, it, ut, threshold, spikes, mode="") -> None:
        c = next(self.colors)["color"]
        self.ax1.plot(it, label = "I & U " + mode, color=c)
        self.ax2.plot(ut, color=c)
        c = next(self.colors)["color"]
        self.ax2.hlines(threshold, 0, len(ut), label = "threshold " + mode, colors = c)
        c = next(self.colors)["color"]
        self.ax2.scatter(spikes, [threshold]*len(spikes), label = "spikes " + mode, color = c)
        return


    def plot_fi_init(self) -> None:
        self.reset()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('I(t)')
        self.ax.set_ylabel('f=1/T')
        self.ax.set_title("Frequency per current")
        self.ax.grid(True)
        return


    def plot_fi_update(self, spikes) -> None:
        self.ax.plot(spikes, color = "blue")
        plt.pause(0.005)
        self.figure.canvas.draw()
        return

    
    def plot_population_activity_init(self, time) -> None:
        self.reset()
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.set_ylabel('number of spikes')
        self.ax1.set_xlabel('time')
        self.ax1.set_title("Population Activity")
        self.ax1.grid(True)
        self.ax1.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax1.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_xlabel('time')
        self.ax2.set_ylabel('index of neuron')
        self.ax2.set_title("Raster Plot")
        self.ax2.grid(True)
        self.ax2.set_xlim([-50,time+50])
        self.ax2.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax2.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_ylabel('I(t)')
        self.ax3.set_xlabel('time')
        self.ax3.set_title("Current per second")
        self.ax3.grid(True)
        self.ax3.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax3.set_xticklabels(np.arange(time+1, dtype=int))
        self.figure.tight_layout(pad=0.5)
        return

    def plot_population_activity_update(self, spikes, it, start_idx=0, mode="") -> None:
        c = next(self.colors)["color"]
        self.ax1.plot(sum(spikes), label="population "+mode, color = c)
        sf = np.flipud(spikes)
        args = np.argwhere(sf)
        self.ax2.scatter(args.T[1,:], start_idx+args.T[0,:], c = c, s=0.5)
        self.ax3.plot(it, color = c, alpha=0.05)
        self.ax3.plot(it.mean(axis=1), color = c)
        return

    def plot_learning_init(self, time) -> None:
        self.reset()
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.set_ylabel('number of neuron')
        self.ax1.set_xlabel('time')
        self.ax1.set_title("Encoder Raster Plot")
        self.ax1.grid(True)
        self.ax1.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax1.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_xlabel('time')
        self.ax2.set_ylabel('index of neuron')
        self.ax2.set_title("Output Population Raster Plot")
        self.ax2.grid(True)
        self.ax2.set_xlim([-50,time+50])
        self.ax2.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax2.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_ylabel('W')
        self.ax3.set_xlabel('time')
        self.ax3.set_title("Weight per second")
        self.ax3.grid(True)
        self.ax3.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax3.set_xticklabels(np.arange(time+1, dtype=int))
        self.figure.tight_layout(pad=0.5)
        return

    def plot_learning_update(self, encoded, spikes, w, mode="") -> None:
        c = next(self.colors)["color"]
        sf = np.flipud(encoded)
        args = np.argwhere(sf)
        self.ax1.scatter(args.T[1,:], args.T[0,:], c = c, s=0.5)
        sf = np.flipud(spikes)
        args = np.argwhere(sf)
        self.ax2.scatter(args.T[1,:], args.T[0,:], c = c, label="population "+mode, s=10)
        self.ax3.plot(w, color = c, alpha=0.1)
        return

    def plot_Rlearning_init(self, time) -> None:
        self.reset()
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.set_ylabel('number of neuron')
        self.ax1.set_xlabel('time')
        self.ax1.set_title("Encoder Raster Plot")
        self.ax1.grid(True)
        self.ax1.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax1.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_xlabel('time')
        self.ax2.set_ylabel('index of neuron')
        self.ax2.set_title("Output Populations Raster Plot")
        self.ax2.grid(True)
        self.ax2.set_xlim([-50,time+50])
        self.ax2.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax2.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_ylabel('activity')
        self.ax3.set_xlabel('time')
        self.ax3.set_title("Activity")
        self.ax3.grid(True)
        self.ax3.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax3.set_xticklabels(np.arange(time+1, dtype=int))
        self.figure.tight_layout(pad=0.5)
        return

    def plot_Rlearning_update(self, encoded, spikes, a, start_idx=0, mode="") -> None:
        c = next(self.colors)["color"]
        sf = np.flipud(encoded)
        args = np.argwhere(sf)
        self.ax1.scatter(args.T[1,:], args.T[0,:], c = c, s=0.5)
        sf = np.flipud(spikes)
        args = np.argwhere(sf)
        self.ax2.scatter(args.T[1,:], start_idx+args.T[0,:], c = c, label="population "+mode, s=10)
        self.ax3.plot(a, color = c)
        return


    def plot_three_population_activity_init(self, time) -> None:
        self.reset()
        self.ax1 = self.figure.add_subplot(331)
        self.ax1.set_ylabel('number of spikes')
        self.ax1.set_xlabel('time')
        self.ax1.set_title("Population Activity")
        self.ax1.grid(True)
        self.ax1.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax1.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax2 = self.figure.add_subplot(334)
        self.ax2.set_xlabel('time')
        self.ax2.set_ylabel('index of neuron')
        self.ax2.set_title("Raster Plot")
        self.ax2.grid(True)
        self.ax2.set_xlim([-50, time+50])
        self.ax2.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax2.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax3 = self.figure.add_subplot(337)
        self.ax3.set_ylabel('I(t)')
        self.ax3.set_xlabel('time')
        self.ax3.set_title("Current per second")
        self.ax3.grid(True)
        self.ax3.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax3.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax4 = self.figure.add_subplot(332)
        self.ax4.set_ylabel('number of spikes')
        self.ax4.set_xlabel('time')
        self.ax4.set_title("Population Activity")
        self.ax4.grid(True)
        self.ax4.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax4.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax5 = self.figure.add_subplot(335)
        self.ax5.set_xlabel('time')
        self.ax5.set_ylabel('index of neuron')
        self.ax5.set_title("Raster Plot")
        self.ax5.grid(True)
        self.ax5.set_xlim([-50,time+50])
        self.ax5.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax5.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax6 = self.figure.add_subplot(338)
        self.ax6.set_ylabel('I(t)')
        self.ax6.set_xlabel('time')
        self.ax6.set_title("Current per second")
        self.ax6.grid(True)
        self.ax6.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax6.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax7 = self.figure.add_subplot(333)
        self.ax7.set_ylabel('number of spikes')
        self.ax7.set_xlabel('time')
        self.ax7.set_title("Population Activity")
        self.ax7.grid(True)
        self.ax7.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax7.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax8 = self.figure.add_subplot(336)
        self.ax8.set_xlabel('time')
        self.ax8.set_ylabel('index of neuron')
        self.ax8.set_title("Raster Plot")
        self.ax8.grid(True)
        self.ax8.set_xlim([-50,time+50])
        self.ax8.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax8.set_xticklabels(np.arange(time+1, dtype=int))

        self.ax9 = self.figure.add_subplot(339)
        self.ax9.set_ylabel('I(t)')
        self.ax9.set_xlabel('time')
        self.ax9.set_title("Current per second")
        self.ax9.grid(True)
        self.ax9.set_xticks(np.arange(time+1, dtype=int)*100)
        self.ax9.set_xticklabels(np.arange(time+1, dtype=int))
        self.figure.tight_layout(pad=0.5)

        return
    
    def plot_three_population_activity_update(self, spikes1, it1, spikes2, it2, spikes3, it3, n1="", n2="", n3="") -> None:
        c1 = next(self.colors)["color"]
        c2 = next(self.colors)["color"]
        c3 = next(self.colors)["color"]
        self.ax1.plot(sum(spikes1), label="population "+n1, color = c1)
        self.ax4.plot(sum(spikes2), label="population "+n2, color = c2)
        self.ax7.plot(sum(spikes3), label="population "+n3, color = c3)
        sf = np.flipud(spikes1)
        args = np.argwhere(sf)
        self.ax2.scatter(args.T[1,:], args.T[0,:], c = c1, s=0.1)
        sf = np.flipud(spikes2)
        args = np.argwhere(sf)
        self.ax5.scatter(args.T[1,:], args.T[0,:], c = c2, s=0.1)
        sf = np.flipud(spikes3)
        args = np.argwhere(sf)
        self.ax8.scatter(args.T[1,:], args.T[0,:], c = c3, s=0.1)
        self.ax3.plot(it1, color = c1, alpha=0.05)
        self.ax3.plot(it1.mean(axis=1), color = c1)
        self.ax6.plot(it2, color = c2, alpha=0.05)
        self.ax6.plot(it2.mean(axis=1), color = c2)
        self.ax9.plot(it3, color = c3, alpha=0.05)
        self.ax9.plot(it3.mean(axis=1), color = c3)
        return


    def plot_encoding_decoding(self, spikes, encoded, decoded):
        self.reset()
        self.ax1 = self.figure.add_subplot(221)
        self.ax1.imshow(spikes)
        self.ax2 = self.figure.add_subplot(2,2,(2,4))
        sf = np.flipud(encoded.T)
        args = np.argwhere(sf)
        self.ax2.scatter(args.T[1,:], args.T[0,:], s=0.1)
        self.ax3 = self.figure.add_subplot(223)
        self.ax3.imshow(decoded)

    def plot_v1(self, image, kernel, output):
        self.reset()
        self.ax1 = self.figure.add_subplot(131)
        self.ax1.set_title("Grayscale Input Image")
        self.ax1.imshow(image)
        self.ax2 = self.figure.add_subplot(132)
        self.ax2.set_title("Kernel")
        self.ax2.imshow(kernel)
        self.ax3 = self.figure.add_subplot(133)
        self.ax3.set_title("Output Image")
        self.ax3.imshow(output)

    def plot_kernel_surface(self, matrix):
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, matrix, cmap=cm.Spectral_r)
        fig.colorbar(surf, shrink=.5, aspect=5)
        plt.show()
