export PYTHONPATH=$PYTHONPATH:./
bash ./tools/dist_test.sh \
        projects/configs/sparse4dv2_r50_HInf_256x704.py \
        ckpts/sparse4dv2_r50_HInf_256x704.pth \
        8