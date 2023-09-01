# for ids in 0 1 2 3 4 5 6 7 8 9
for ids in 2 5 8 9
do 
    # python demo.py --dataset wikipedia --ratio_ids $ids --epoch 120 --device cuda:0
    python demo.py --dataset wikipedia --ratio_ids $ids --epoch 100 --device cuda:2
done