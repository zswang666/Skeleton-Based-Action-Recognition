#########################################################################################
##  Please copy this code and utils.py to your openpose directory and execute that one ##
#########################################################################################
import os
import sys
import tqdm
import subprocess
import pdb

from utils import *

WRITE_IMAGES = True

def main():
    openpose_dir = os.path.expanduser(sys.argv[1])
    openpose_bin = os.path.join(openpose_dir,"build/examples/openpose/openpose.bin")
    #extract_kth(openpose_bin)
    extract_weizmann(openpose_bin)

def extract_kth(openpose_bin):
    input_root_dir = validate_dir("/home/Dataset/KTH/frames", auto_mkdir=False)
    output_root_dir = validate_dir("/home/Dataset/KTH/openpose")
    #output_root_dir = validate_dir("/home/johnson/Desktop/kth_openpose")

    for cname in os.listdir(input_root_dir): # iterate class
        cpath = os.path.join(input_root_dir,cname)
        person_pbar = tqdm.tqdm(sorted(os.listdir(cpath)))
        for pname in person_pbar: # iterate person
            person_pbar.set_description("Processing {}".format(pname))
            ppath = os.path.join(cpath,pname)
            for tname in os.listdir(ppath): # iterate trial
                tpath = os.path.join(ppath,tname)
                json_path = validate_path(output_root_dir,"keypoints",cname,pname,tname)
                if WRITE_IMAGES:
                    imgs_path = validate_dir(output_root_dir,"images",cname,pname,tname)
                    command = openpose_bin + \
                              " --image_dir {} \
                                --no_display \
                                --write_json {} \
                                --write_images {}".format(tpath,json_path,imgs_path)
                else:
                    command = openpose_bin + \
                              " --image_dir {} \
                                --no_display \
                                --write_json {}".format(tpath,json_path)
                subprocess.call(command, shell=True)

def extract_weizmann(openpose_bin):
    input_root_dir = validate_dir("/home/Dataset/WEIZMANN/frames", auto_mkdir=False)
    #output_root_dir = validate_dir("/home/Dataset/WEIZMANN/openpose")
    output_root_dir = validate_dir("/home/johnson/Desktop/weizemann_openpose")

    for cname in os.listdir(input_root_dir): # iterate class
        cpath = os.path.join(input_root_dir,cname)
        person_pbar = tqdm.tqdm(sorted(os.listdir(cpath)))
        for pname in person_pbar: # iterate person
            person_pbar.set_description("Processing {}".format(pname))
            ppath = os.path.join(cpath,pname)
            json_path = validate_path(output_root_dir,"keypoints",cname,pname)
            if WRITE_IMAGES:
                imgs_path = validate_dir(output_root_dir,"images",cname,pname)
                command = openpose_bin + \
                          " --image_dir {} \
                            --no_display \
                            --write_json {} \
                            --write_images {}".format(ppath,json_path,imgs_path)
            else:
                command = openpose_bin + \
                          " --image_dir {} \
                            --no_display \
                            --write_json {}".format(ppath,json_path)
            subprocess.call(command, shell=True)

if __name__=="__main__":
    main()
