import torch
import numpy as np
import os
import argparse
from CPCMR import Solver
import time


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    seeds = 1
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nus-wide', help='Dataset name: wikipedia, nus-wide, pascal, xmedianet')
    parser.add_argument('--ratio_ids', type=int, default=4, help='0-9')
    # parser.add_argument('--type', type=str, default='CLIP', help='PAN-M/CLIP')
    parser.add_argument('--epoch', type=int, default='50', help='default:50 epochs')
    parser.add_argument('--device', type=str , default="cuda:0", help='cuda device')
    config = parser.parse_args()

    start_time = time.time()
    Final_mean_MAP = {}
    Final_std_MAP = {}

    data_ratios=[[50,50,0], [50,0,50], [50,25,25], [30,70,0], [30,0,70], [30,35,35], [10,90,0], [10,0,90], [10,45,45], [100,0,0]]      # [dual data, only image, only text]
    data_ratio = data_ratios[config.ratio_ids]
    
    config.data_ratio = data_ratio
    total_map = []
    map_i2t = []
    map_t2i = []
    print('=============== {}--{}--Total epochs:{} ==============='.format(config.dataset, data_ratio, config.epoch))
    # for state in [1,2,3,4,5]:
    for state in [1]:    
        print('...Training is beginning...',state)
        solver = Solver(config)
        map, i2t, t2i = solver.train()
        total_map.append(map)
        map_i2t.append(i2t)
        map_t2i.append(t2i)

    mean_map = round(np.mean(total_map), 4)
    std_map = round(np.std(total_map), 4)
    i2t_map = round(np.mean(map_i2t),4)
    t2i_map = round(np.mean(map_t2i),4)

    print("mean_map:{0}, std_map:{1} , i2t:{2}, t2t:{3} .".format(mean_map,std_map,i2t_map,t2i_map))

    time_elapsed = time.time() - start_time
    print(f'Total Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')

    with open('result.txt', 'a+', encoding='utf-8') as f:
        f.write("[{0}-{1} epochs]".format(config.dataset, config.epoch))
        f.write("{0}: mean:{1}, std:{2},i2t:{3}, t2t:{4}.\n".format(data_ratio, mean_map, std_map,i2t_map,t2i_map))
        # f.write("--------------------------------------------\n")
    