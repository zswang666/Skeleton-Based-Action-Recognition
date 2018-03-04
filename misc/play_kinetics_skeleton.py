# sys
import os
import sys
import numpy as np
import json
import pprint

# debug
import ipdb

data_path = "../datasets/kinetics-skeleton/kinetics_val"
label_path = "../datasets/kinetics-skeleton/kinetics_val_label.json"

# load file
fnames = os.listdir(data_path)
with open(label_path, 'r') as f:
    label_info = json.load(f)

# look at label file
print("##############################")
print("# of labels = {}".format(len(label_info)))
print("Type of labels = {}".format(type(label_info)))
for idx, lname in enumerate(label_info.keys()):
    print("label_info[{}] = {}".format(lname, label_info[lname]))
    if idx>=1:
        print("only showing 2 labels")
        break

# look at data
print("##############################")
print("Showing one piece of data")
for idx, fname in enumerate(fnames):
    fpath = os.path.join(data_path, fname)
    with open(fpath, 'r') as f:
        video_info = json.load(f)

    # look at video_info
    print("# = {}".format(len(video_info)))
    print("type = {}".format(type(video_info)))
    print("keys = {}".format(video_info.keys()))
    print("\"data\" type = {}, # = {}".format(type(video_info["data"]), len(video_info["data"])))
    print("\"label\" = {}".format(video_info["label"]))
    print("\"label_index\" = {}".format(video_info["label_index"]))

    # look at video_info["data"]
    print("")
    print("look at one_data=video_info[\"data\"][\"0\"]")
    one_data = video_info["data"][0]
    pprint.pprint(one_data)

    ipdb.set_trace()

    if idx>=100:
        print("only showing 1 piece of data")
        break
