export PYTHONPATH=/home/birk/Projects/robustSD/

python train.py --img 512 --batch 16 --epochs 300 --patience 20 --data $1 --weights yolov5m.pt
