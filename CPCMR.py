from distutils.command.config import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluate import fx_calc_map_label,fx_calc_recall
from load_dataset import  load_dataset
from metric import Cluster, Semantic_loss
from model import FuseTransEncoder, ImageTransEncoder, TextTransEncoder, Prototype
import scipy.io as sio


class Solver(object):
    def __init__(self, config):
        self.batch_size = 128  # 128
        #########################################

        self.data_ratio= config.data_ratio
        self.total_epoch = config.epoch
        #########################################
        self.dataset  = config.dataset

        if config.dataset == 'nus-wide':
            self.num_class = 10
        elif config.dataset == 'pascal':
            self.num_class = 20
        elif config.dataset == 'wikipedia':
            self.num_class = 10
        elif config.dataset == 'xmedianet':
            self.num_class = 200
        else:
            print('ERROR!')

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        num_layers, self.token_size, nhead = 2, 2048, 2
        anchor_dim = int(self.token_size/2) 

        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead, self.num_class).to(self.device)
        self.ImageTrans = ImageTransEncoder(num_layers, int(self.token_size/2), nhead,self.num_class).to(self.device)
        self.TextTrans = TextTransEncoder(num_layers, int(self.token_size/2), nhead,self.num_class).to(self.device)
        self.Pro = Prototype(anchor_dim, self.num_class, self.device ).to(self.device)
        
        paramsFuse_to_update = list(self.FuseTrans.parameters()) 
      
        paramImageTrans = list(self.ImageTrans.parameters())
        paramTextTrans = list(self.TextTrans.parameters())

        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_ImageTrans = optim.Adam(paramImageTrans, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextTrans = optim.Adam(paramTextTrans, lr=1e-3, betas=(0.5, 0.999))

        self.train_idx, self.len_test, self.dual_size, self.oimg_size, self.otext_size, self.batch_len,  self.train_data, self.test_data = load_dataset(self.dataset, self.batch_size, self.data_ratio)
      
        self.MSE_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.dual_idx, self.oimg_idx, self.otext_idx = self.train_idx[0], self.train_idx[1], self.train_idx[2]
    
        

    def train(self):
        trainning_loss = []
        
        for epoch in range(self.total_epoch):
            np.random.shuffle(self.dual_idx)
            np.random.shuffle(self.oimg_idx)
            np.random.shuffle(self.otext_idx)
            train_loss = self.trainstep()
          
            if(epoch%10==0):
                img2text, text2img = self.evaluate() 
                print('I2T:',img2text, ', T2I:',text2img)

        #     if(epoch%50==0):
        #         trainning_loss.append(train_loss)
        # print(trainning_loss)
        # img2text, text2img = self.evaluate() 
        # print('I2T:',img2text, ', T2I:',text2img)
        # i_map.append(img2text)
        # t_map.append(text2img)

        return (img2text + text2img) / 2.,img2text,text2img
      
    def evaluate(self):
        self.ImageTrans.eval() 
        self.TextTrans.eval()
        self.FuseTrans.eval()

        test_data = torch.tensor(self.test_data.reshape(-1,self.batch_size,2049),dtype= torch.float32).to(self.device)
       
    
        test_img = test_data[:,:,:1024]
        test_text = test_data[:,:,1024:2048]
        test_label = test_data[:,:,-1].type(torch.int64).reshape(-1)
        # test = test_data[:,:,:-1]
       
        with torch.no_grad():

            anchor = self.Pro()
            only_image_recon = self.ImageTrans(test_img, anchor)
            only_text_recon = self.TextTrans(test_text, anchor)

            only_image_recon = only_image_recon.reshape(-1,1024)
            only_text_recon = only_text_recon.reshape(-1,1024)
            # only_image_recon, only_text_recon= self.FuseTrans(test)
            # only_image_recon = only_image_recon.reshape(-1,1024)
            # only_text_recon = only_text_recon.reshape(-1,1024)
            
            te_imgs = only_image_recon.cpu().numpy()
            te_txts = only_text_recon.cpu().numpy()
            te_labels = test_label.cpu().numpy()
        
        te_imgs = te_imgs[:self.len_test,:]  # for visualization
        te_txts = te_txts[:self.len_test,:]  # for visualization
        te_labels = te_labels[:self.len_test] 
        # filename= 'test_data'
        # np.savez(filename,te_imgs= te_imgs,te_txts=te_txts,te_labels=te_labels)


        i_map = fx_calc_map_label(te_imgs, te_txts, te_labels)
        t_map = fx_calc_map_label(te_txts, te_imgs, te_labels)


        # ord_i2t, _ = fx_calc_recall(self,te_imgs, te_txts, te_labels)
        # ord_t2i, label_matrix = fx_calc_recall(self,te_txts, te_imgs, te_labels)
        # _dict = {
        #     'ord_i2t':ord_i2t,
        #     'ord_t2i':ord_t2i,
        #     'label_matrix': label_matrix
        # }
        # savepath = 'matlab/Data/ours/' + str(self.data_ratio) + self.dataset+".mat"
        # sio.savemat(savepath,_dict)

        return i_map, t_map 
    
    def trainstep(self):
        self.FuseTrans.train()
        self.ImageTrans.train() 
        self.TextTrans.train()
    
        running_loss = 0.0
    
        Fuse_tokens = torch.zeros((self.batch_len, self.batch_size, 2049), dtype= torch.float32).to(self.device)
        # Fuse_tokens = torch.zeros((self.batch_len, self.dual_size, 2049), dtype= torch.float32).to(self.device)
     
        for batch_idx in range(self.batch_len):
            small_idx_dual = self.dual_idx[batch_idx * self.dual_size: (batch_idx + 1) * self.dual_size]
         
            small_idx_img = self.oimg_idx[batch_idx * self.oimg_size: (batch_idx + 1) * self.oimg_size]
            small_idx_txt = self.otext_idx[batch_idx * self.otext_size: (batch_idx + 1) * self.otext_size]

            
            train_dual_data = torch.tensor(self.train_data[small_idx_dual]).to(self.device)
            train_only_img = torch.tensor(self.train_data[small_idx_img]).to(self.device)
            train_only_text = torch.tensor(self.train_data[small_idx_txt]).to(self.device)
           

            temp_img = torch.cat([train_dual_data, train_only_img],0)
            img_mask = -torch.mean(temp_img[:,:1024])
           
            # img_mask = 100
            temp_text = torch.cat([train_dual_data, train_only_text],0)
            text_mask = -torch.mean(temp_text[:,1024:2048])
           
            # text_mask = 100


            train_only_img[:,1024:2048] = text_mask
            train_only_text[:, :1024] = img_mask
            temp_tokens = torch.cat([train_dual_data, train_only_img, train_only_text],0)
            # temp_tokens = train_dual_data
            # print(temp_tokens.shape)
            Fuse_tokens[batch_idx] = temp_tokens
       
        if self.batch_len <= 20:   # 小数据集 20 wikipedia ,pascal
      
            loss = 0.0
            Img_tokens = Fuse_tokens[:, :, :1024]
            Text_tokens = Fuse_tokens[:, :, 1024:2048]
            labels = Fuse_tokens[:,:,-1].type(torch.int64)
            labels = labels.reshape(-1)
            
            img_embedding, text_embedding, cls_out_image, cls_out_text = self.FuseTrans(Fuse_tokens[:,:,:-1])
            img_embedding = img_embedding.reshape(-1, 1024)
            text_embedding = text_embedding.reshape(-1,1024)

            
           
            
            anchor = self.Pro() 
            Cluster_loss = Cluster(img_embedding, anchor, labels) + Cluster(text_embedding, anchor, labels) 
            semantic_loss = Semantic_loss(cls_out_image, cls_out_text, labels)

            text_recon = self.ImageTrans(Img_tokens, anchor)
            img_recon = self.TextTrans(Text_tokens, anchor)
            img_recon = img_recon.reshape(-1,1024)
            text_recon = text_recon.reshape(-1,1024)
         
            img_loss = F.kl_div(img_recon.log_softmax(dim=-1), text_embedding.softmax(dim=-1), reduction='batchmean')
            text_loss = F.kl_div(text_recon.log_softmax(dim=-1), img_embedding.softmax(dim=-1), reduction='batchmean')
            recon_loss = img_loss + text_loss
            
            loss =  semantic_loss + Cluster_loss + recon_loss 
            
            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageTrans.zero_grad() 
            self.optimizer_TextTrans.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageTrans.step()
            self.optimizer_TextTrans.step()
           
            running_loss += loss.item()
           

        else:   # 针对大数据集  nus-wide, xmedianet
            batch_len = 21
            
            if(Fuse_tokens.shape[0] % 21 !=0):  # xmedianet
                Fuse_tokens = torch.cat([Fuse_tokens, Fuse_tokens[0].unsqueeze(0)], 0)  
            batch_nums = int(Fuse_tokens.shape[0]/batch_len)
           
            Img_tokens = Fuse_tokens[:, :, :1024]
            Text_tokens = Fuse_tokens[:, :, 1024:2048]
            labels = Fuse_tokens[:,:,-1].type(torch.int64)
            
            for i in range(0, batch_nums):
                loss = 0.0
                start_idx = i*batch_len
                stop_idx = (i+1)*batch_len

                Image_batch = Img_tokens[start_idx:stop_idx]
                Text_batch = Text_tokens[start_idx:stop_idx]
                img_embedding, text_embedding, cls_out_image, cls_out_text = self.FuseTrans(torch.cat([Image_batch,Text_batch],2))
                L = labels[start_idx:stop_idx]
                Lable = L.reshape(-1)
                img_embedding = img_embedding.reshape(-1,1024)
                text_embedding = text_embedding.reshape(-1,1024)

               
                
                anchor = self.Pro()
                Cluster_loss = Cluster(img_embedding, anchor, Lable) + Cluster(text_embedding, anchor, Lable)

                semantic_loss = Semantic_loss(cls_out_image,cls_out_text, Lable) 
                text_recon = self.ImageTrans(Image_batch, anchor)
                img_recon = self.TextTrans(Text_batch, anchor)
                img_recon = img_recon.reshape(-1,1024)
                text_recon = text_recon.reshape(-1,1024)
                
                img_loss = F.kl_div(img_recon.log_softmax(dim=-1), text_embedding.softmax(dim=-1), reduction='batchmean')
                text_loss = F.kl_div(text_recon.log_softmax(dim=-1), img_embedding.softmax(dim=-1), reduction='batchmean')
                recon_loss = img_loss + text_loss
                loss =   semantic_loss + Cluster_loss + recon_loss 
                
                self.optimizer_FuseTrans.zero_grad()
                self.optimizer_ImageTrans.zero_grad() 
                self.optimizer_TextTrans.zero_grad()
                loss.backward()
                self.optimizer_FuseTrans.step()
                self.optimizer_ImageTrans.step()
                self.optimizer_TextTrans.step()
                running_loss += loss.item()
       
        return running_loss

