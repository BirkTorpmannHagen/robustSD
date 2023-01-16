#for (( i = 0; i < 10; i++ )); do
#  samples=( $(seq 2 6 | shuf) )
#  python ../val.py --img 512 --data $1 --weights "../runs/train/exp${samples[1]}/weights/best.pt" "../runs/train/exp${samples[2]}/weights/best.pt" "../runs/train/exp${samples[3]}/weights/best.pt" "../runs/train/exp${samples[4]}/weights/best.pt" "../runs/train/exp${samples[5]}/weights/best.pt"  &>> "vanilla_ensemble_pca_fold3.txt"
#done

#for (( i = 0; i < 10; i++ )); do
#  samples=( $(seq 31 40 | shuf) )
#  python ../val.py --img 512 --data $1 --weights "../runs/train/exp${samples[1]}/weights/best.pt" "../runs/train/exp${samples[2]}/weights/best.pt" "../runs/train/exp${samples[3]}/weights/best.pt" "../runs/train/exp${samples[4]}/weights/best.pt" "../runs/train/exp${samples[5]}/weights/best.pt"  &>> "vanilla_ood.txt"
#done
#
#for (( i = 0; i < 10; i++ )); do
#  samples=( $(seq 41 50 | shuf) )
#  python ../val.py --img 512 --data $1 --weights "../runs/train/exp${samples[1]}/weights/best.pt" "../runs/train/exp${samples[2]}/weights/best.pt" "../runs/train/exp${samples[3]}/weights/best.pt" "../runs/train/exp${samples[4]}/weights/best.pt" "../runs/train/exp${samples[5]}/weights/best.pt"  &>> "batch_select_ensemble_ood.txt"
#done

for (( i = 0; i < 10; i++ )); do
  samples=( $(seq 26 32 | shuf) )
  python ../val.py --img 512 --data $1 --weights "../runs/train/exp${samples[1]}/weights/best.pt" "../runs/train/exp${samples[2]}/weights/best.pt" "../runs/train/exp${samples[3]}/weights/best.pt" "../runs/train/exp${samples[4]}/weights/best.pt" "../runs/train/exp${samples[5]}/weights/best.pt"  &>> "batch_select_clustered_ensemble_ood_no_compress.txt"
done