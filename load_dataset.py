from torch.utils.data.dataset import Dataset
import numpy as np
import math
import pickle

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs):
        self.images = images
        self.texts = texts
        self.labs = labs
   

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        return img, text, lab

    def __len__(self):
        count = len(self.texts)
        return count



def prepared_data(imgs,texts,labs, bsz):
    
    train_lens =len(labs['train'])
    test_lens = len(labs['test'])
    train_fill_lens = bsz -(train_lens % bsz)
    test_fill_lens = bsz- (test_lens % bsz)
    if (train_fill_lens!=0) :  # 填充训练集数据
        imgs['train'] = np.concatenate((imgs['train'], imgs['train'][:train_fill_lens]), axis= 0)
        texts['train'] = np.concatenate((texts['train'], texts['train'][:train_fill_lens]), axis= 0)
        labs['train'] = np.expand_dims(np.concatenate((labs['train'], labs['train'][:train_fill_lens]), axis= 0),axis=1)
        train_data = np.concatenate((imgs['train'], texts['train'],labs['train']),axis= 1)

    if (test_fill_lens!=0):   # 填充测试集数据
        imgs['test'] = np.concatenate((imgs['test'], imgs['test'][:test_fill_lens]), axis= 0)
        texts['test'] = np.concatenate((texts['test'], texts['test'][:test_fill_lens]), axis= 0)
        labs['test'] = np.expand_dims(np.concatenate((labs['test'], labs['test'][:test_fill_lens]), axis= 0), axis=1)
        test_data = np.concatenate((imgs['test'], texts['test'], labs['test']),axis=1)
    return train_data ,test_data


def load_dataset(dataset, batch_size, data_ratio):
    
    train_loc = 'data/'+ dataset +'/clip_train.pkl'
    test_loc = 'data/'+ dataset +'/clip_test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
    
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']   
        test_texts = data['text']      
        test_images = data['image']   
    
    if dataset == 'pascal':
        train_labels = train_labels - 1
        test_labels = test_labels - 1

    imgs = {'train': train_images, 'test': test_images}
    texts = {'train': train_texts,  'test': test_texts}
    labs = {'train': train_labels, 'test': test_labels}
    print(labs['train'].shape,labs['test'].shape)
    train_data, test_data = prepared_data(imgs, texts, labs, batch_size)

    train_size = train_data.shape[0]
    # test_size = config.test_data.shape[0]
    batch_len = int(train_size/batch_size)

    total_dual_size, total_oimg_size = math.ceil(batch_size * data_ratio[0]*0.01)* batch_len, math.floor(batch_size * data_ratio[1]*0.01)* batch_len
    total_otext_size = train_size - total_dual_size - total_oimg_size
    idx_train = np.random.permutation(train_size)
    
    train_dual_idx = idx_train[:total_dual_size]
    train_oimg_idx = idx_train[total_dual_size: total_dual_size+ total_oimg_size]
    train_otext_idx = idx_train[total_dual_size+ total_oimg_size:]

    dual_size = int(total_dual_size/batch_len)
    oimg_size = int(total_oimg_size/batch_len)
    otext_size = int(total_otext_size / batch_len)
    
    return  (train_dual_idx,train_oimg_idx,train_otext_idx), len(test_labels), dual_size, oimg_size, otext_size,batch_len, train_data, test_data


       


    
  
        