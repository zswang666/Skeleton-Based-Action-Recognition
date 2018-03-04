import os
import argparse
import pickle 
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from utils import *

def main():
    # arguments
    parser = get_parser()
    p = parser.parse_args()
    n_primitives = 5 #TODO should be a real argument
    fps = 25
    output_dir = validate_dir(p.output_dir)

    # fetch visualization data
    vis_data = pickle.load(open(p.pkl_path,"rb"))
    ps_list = vis_data["primitives_seq_list"]
    data_list = vis_data["data_list"]

    # define colormap
    cmap = plt.get_cmap('PiYG', n_primitives+1)
    cmap_list = [cmap(i)[:3] for i in range(cmap.N)]
    cmap_list[0] = (0.,0.,0.)
    cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)

    fig, axs = plt.subplots(2,1)
    cb = colorbar_index(ncolors=n_primitives+1, cmap=cmap, orientation="horizontal")
    for i, pdata_path in enumerate(data_list):
        # modify pdata_path from openpose to frames
        tmp = pdata_path.split('/')
        pdata_path = os.path.join(*tmp[0:3],"frames",*tmp[5:])
        cls = tmp[5]
        sample_name = tmp[6]

        # frame path names
        frame_names = sorted(os.listdir(pdata_path))
        
        # movie writer
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=pdata_path, artist="Johnson",
                        comment="hey yo!!")
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        # axes0 for video
        axs[0].axis("off")
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_title(sample_name)

        # axes1 for transition visualization
        mat = np.expand_dims(np.array(ps_list[i]), axis=0) + 1
        cax = axs[1].imshow(mat, cmap=cmap, vmin=0, vmax=n_primitives, extent=[0,mat.shape[1],0,1])
        l, = axs[1].plot([], [], 'k-o') # scanning line

        line_x = [0,0]
        line_y = [0,1]
        save_dir = validate_dir(os.path.join(output_dir,cls))
        save_path = os.path.join(save_dir,"transition_"+sample_name+".mp4")
        with writer.saving(fig, save_path, 100):
            scanning_step = float(mat.shape[1]) / len(frame_names)
            for frame_name in frame_names:
                # show video
                axs[0].clear()
                frame_path = os.path.join(pdata_path,frame_name)
                frame = Image.open(frame_path)
                axs[0].imshow(frame, aspect="auto")

                # scanning line
                line_x[0] += scanning_step #TODO: seems like something's wrong
                line_x[1] += scanning_step #TODO
                l.set_data(line_x, line_y)

                writer.grab_frame()

        # clear vis
        axs[0].clear()
        axs[1].clear()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    return parser

if __name__=="__main__":
    main()
