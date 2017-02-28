export CUDA_VISIBLE_DEVICES=2,3
#export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES
export CITYSCAPES_DATASET=/mnt/sdc1/shangxuan/dataset/cityscape
echo $CITYSCAPES_DATASET
python main.py
