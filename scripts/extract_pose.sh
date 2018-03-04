OPENPOSE_DIR=/home/johnson/Desktop/workspace/FGAR/openpose
cp ../tool/extract_pose.py ${OPENPOSE_DIR}
cp ../tool/utils.py ${OPENPOSE_DIR}
cd ${OPENPOSE_DIR}
python extract_pose.py ${OPENPOSE_DIR}
cd -
