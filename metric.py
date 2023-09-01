import torch
import torch.nn.functional as F


def Cluster(features, centers, labels):
   
    distance = (features - centers[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]
    return distance


def Semantic_loss(cls_out_img, cla_out_text, label):
    output1 = F.log_softmax(cls_out_img, dim=1)
    
    loss1 = F.nll_loss(output1, label)
    # print('cls_out_img:',cls_out_img.shape, "label:",label.shape, "log_sofmax:",output1.shape, "nll_loss:",loss1.shape)
    
    output2 = F.log_softmax(cla_out_text, dim=1)
    loss2 = F.nll_loss(output2, label)
    return loss1+loss2

