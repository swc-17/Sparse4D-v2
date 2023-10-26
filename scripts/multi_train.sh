export PYTHONPATH=$PYTHONPATH:./
# bash ./tools/dist_train.sh \
#     projects/configs/sparse4dv2_r50_HInf_256x704.py \
#     1
bash ./tools/dist_train.sh \
    projects/configs/sparse4d_r101_H1.py \
    1