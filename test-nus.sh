# for ids in 0 1 2 3 4 5 6 7 8 9
for ids in 2 5 8 9
do 
    python demo.py --dataset nus-wide --ratio_ids $ids --epoch 80 --device cuda:1
done