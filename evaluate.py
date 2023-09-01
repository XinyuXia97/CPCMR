import numpy as np
import scipy.spatial
import scipy.io as sio

def fx_calc_map_label(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = dist.argsort() # [batch, batch]
   
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def fx_calc_recall(self, image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = (dist.argsort() + 1).T # [batch, batch]
   
   
    label_matrix = np.zeros((label.shape[0],label.shape[0]))
 
    # for i in range(label.shape[0]):
    #     index = np.where(label[i] == label)[0]
    #     label_matrix[i][index] = 1\
    for i in range(label.shape[0]):
        for j in range(label.shape[0]):
            if label[i] == label[j]:
                label_matrix[i, j] = 1

    # _dict = {
    #     'ord':ord,
    #     'label_matrix': label_matrix
    # }
    # sio.savemat(str(self.data_ratio)+'pascal.mat',_dict)
    return ord, label_matrix
   


    # ranks = np.zeros(image.shape[0])
    # for i in range(image.shape[0]):
    #     q_label = label[i]
    #     r_labels = label[ord[i]]
    #     ranks[i] = np.where(r_labels == q_label)[0][0]
    # print(ranks)

    # # R@K
    # for i in range(image.shape[0]):
    #     q_label = label[i]
    #     r_labels = label[ord[i]]
      
    #     ranks[i] = np.where(r_labels == q_label)[0][0]
    #     # print(np.where(r_labels == q_label))
      
    # r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Prec@K
    # for K in [1, 2, 4, 8, 16]:
    #     prec_at_k = calc_precision_at_K(ord, label, K)
    #     print("P@{} : {:.3f}".format(k, 100 * prec_at_k))

    # return r1, r5, r10