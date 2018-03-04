import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def test1():
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    x0, y0 = 0, 0

    with writer.saving(fig, "data/writer_test.mp4", 100):
        for i in range(100):
            x0 += 0.1 * np.random.randn()
            y0 += 0.1 * np.random.randn()
            l.set_data(x0, y0)
            writer.grab_frame()

def test2():
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    fps = 15
    n_secs = 10
    writer = FFMpegWriter(fps=fps, metadata=metadata)
   
    fig, axs = plt.subplots(1,1)
    axs.imshow(np.random.uniform(0,1,(1,10)), extent=[0,10,0,1])
    l, = axs.plot([], [], 'k-o')
    
    line_x = [0,0]
    line_y = [0,1]

    with writer.saving(fig, "data/writer_test2.mp4", 100):
        for i in range(fps*n_secs):
            line_x[0] += 1./fps
            line_x[1] += 1./fps
            l.set_data(line_x, line_y)
            writer.grab_frame()
    
def test3():
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    fps = 15
    n_secs = 10
    writer = FFMpegWriter(fps=fps, metadata=metadata)
   
    fig, axs = plt.subplots(2,1)
    
    # axes0
    axs[0].axis('off')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    # axes1
    cax = axs[1].imshow(np.random.uniform(0,1,(1,10)), extent=[0,10,0,1])
    cbar = fig.colorbar(cax, ticks=range(1,11), orientation="horizontal")
    cbar.ax.set_xticklabels(['1','2','3'])
    l, = axs[1].plot([], [], 'k-o')

    line_x = [0,0]
    line_y = [0,1]
    with writer.saving(fig, "data/writer_test3.mp4", 100):
        for i in range(fps*n_secs):
            line_x[0] += 1./fps
            line_x[1] += 1./fps
            l.set_data(line_x, line_y)

            axs[0].clear()
            axs[0].imshow(np.random.uniform(0,1,(10,10)), aspect="auto")
            
            writer.grab_frame()

test3()
