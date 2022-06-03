import streamlit as st
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import numpy as np
import pandas as pd
import os

st.title('Video segmentation Project')
st.header("Video segmentation example")
st.text("Please upload a Video")
VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm","avi"]

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")

def get_video_files_in_dir(directory):
    out = []
    for item in os.listdir(directory):
        try:
            name, ext = item.split(".")
        except:
            continue
        if name and ext:
            if ext in VIDEO_EXTENSIONS:
                out.append(item)
    return out


avdir = os.path.expanduser("test") #le repertoire contenant les videos est nommé 'test'
files = get_video_files_in_dir(avdir)

if len(files) == 0:
    st.write(
        "Put some video files in your home directory (%s) to activate this player."
        % avdir
    )

else:
    filename = st.selectbox(
        "Select a video file from your home directory (%s) to play" % avdir,
        files,
        0,
    )

    st.video(os.path.join(avdir, filename))
print(avdir+" "+filename)
result = st.button('Run classifies/segmented video')
if result:
    gif_runner = st.image("wait.gif")
    ins.process_video(avdir+"/"+filename, show_bboxes=True, frames_per_second=15,output_video_name="output_video_F.mp4")
    gif_runner.empty()
    st.video(os.path.join("output_video_F.mp4"))



