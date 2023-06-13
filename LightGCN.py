# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:01:38 2023

@author: simon
"""
import os
import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.set_random_seed(0)
    tf.compat.v1.reset_default_graph()
except:
    import tensorflow as tf
    tf.random.set_seed(0)
    tf.reset_default_graph()
import pandas as pd
from scipy import sparse as sp
import pickle
from datetime import datetime
os.environ["TF_NUM_INTRAOP_THREADS"] = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(0)
np.random.seed(0)

class data(object):
    def __init__(self):
        self.trainingSetRatio = 0.8
        self.validationSetRatio = 0.
        self.binarize = 1
        self.df = self.IO()
        self.train, self.val, self.test, self.num_users, self.num_items = self.splitSet()
        self.user2id, self.id2user, self.item2id, self.id2item = self.idxId()
        self.user_train_profile, self.user_val_profile, self.user_test_profile = self.userProfile()
        self.R = self.ratingSparseAdj()
        self.joint_R = self.jointSparseAdj()
        
    def IO(self):
        #binarize
        df = pd.read_csv('datasets/LastFM/ratings.txt', sep='\t', header=None)
        df = df[df[2]>=self.binarize]
        #remove the extra useless attributes
        try:
            for i in range(3, len(df.columns)):
                df = df.drop(df.columns[i],axis=1)
        except:
            pass
        return df
    
    def splitSet(self):
        #split training/validation/test set
        train = self.df.sample(frac=self.trainingSetRatio)
        rest = self.df.drop(train.index)
        val = rest.sample(frac=self.validationSetRatio/(1-self.trainingSetRatio))
        test = rest.drop(val.index)
        
        #to numpy array
        train = {'users': train[0].to_numpy(), 'pos_item': train[1].to_numpy()}
        val = {'users': val[0].to_numpy(), 'pos_item': val[1].to_numpy()}
        test = {'users': test[0].to_numpy(), 'pos_item': test[1].to_numpy()}
        
        #total user/item
        num_users = len(np.unique(train['users']))
        num_items = len(np.unique(train['pos_item']))
        return train, val, test, num_users, num_items
    
    def userProfile(self):
        #sum up purchase record for each user
        train_profile, val_profile, test_profile = {}, {}, {}
        for user in self.df[0].unique():
            train_profile[user] = self.train['pos_item'][np.where(self.train['users']==user)]
            val_profile[user] = self.val['pos_item'][np.where(self.val['users']==user)]
            test_profile[user] = self.test['pos_item'][np.where(self.test['users']==user)]
        return train_profile, val_profile, test_profile
    
    ##mapping-only training node
    def idxId(self):
        user2id = {}
        id2user = {}
        item2id = {}
        id2item = {}
        for cnt, elem in enumerate(np.unique(self.train['users'])):
            user2id[elem] = cnt            
            id2user[cnt] = elem
        for cnt, elem in enumerate(np.unique(self.train['pos_item'])):
            item2id[elem] = cnt
            id2item[cnt] = elem
        return user2id, id2user, item2id, id2item
    
    #only training node
    def ratingSparseAdj(self):
        row, col, entries = [], [], []
        for idx in range(self.train['users'].size):
            row += [self.user2id[self.train['users'][idx]]]
            col += [self.item2id[self.train['pos_item'][idx]]]
            entries += [1.0/np.count_nonzero(self.train['users']==self.train['users'][idx])]
        ratingMat = sp.coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMat
    
    #only training node
    def jointSparseAdj(self):
        n_nodes = self.num_users + self.num_items
        row_idx = []
        col_idx = []
        for idx in range(self.train['users'].size):
            row_idx.append(self.user2id[self.train['users'][idx]])
            col_idx.append(self.item2id[self.train['pos_item'][idx]])
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes,n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(axis=1))
        
        #in case divide by zero error
        with np.errstate(divide='ignore'):
            d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

class evaluation(object):
    def predictForRanking(self, prompt_name, epoch=0):
        #define
        if prompt_name == 'validation':
            self.prompt_set = self.val
            self.prompt_profile = self.user_val_profile
        if prompt_name == 'test':
            self.prompt_set = self.test
            self.prompt_profile = self.user_test_profile    
        self.rec_list = {}
        self.hits = {}
        global_mean = self.train['users'].size
        trained_users = np.unique(self.train['users'])
        
        #predict the topk item
        for i, user in enumerate(np.unique(self.prompt_set['users'])):
            if user in trained_users:
                idx = self.user2id[user]
                candidates = self.V.dot(self.U[idx])
            else: #in case this user not in training    
                candidates = np.array([1.0] * self.num_items)
            for j, item in enumerate(self.user_train_profile[user]):
                candidates[self.item2id[item]] = 0.
            #self.rec_list[user] = [self.id2item[elem] for elem in candidates.argsort()[::-1][:self.top_k]] #much slower
            ids, scores = self.find_k_largest(self.top_k, candidates)
            self.rec_list[user] = [self.id2item[iid] for iid in ids]
            #print(ids, scores)
            self.hits[user] = len(set(self.rec_list[user]).intersection(set(self.prompt_profile[user])))
        #metric calculation
        currentMetrics = {'epoch': epoch, 'precision': self.precision(), 'recall': self.recall(), 'NDCG': self.NDCG()}
        currentMetrics['F1'] = self.F1(currentMetrics['precision'], currentMetrics['recall'])
        
        #update
        #if prompt_name = "val"
        flag = 0
        for key in currentMetrics.keys():
            if currentMetrics[key] >= self.bestMetrics[key]:
                flag += 1
        if flag > 2:
            self.bestMetrics = currentMetrics
        
        #output metrics
        print("Evaluation:")
        print("------------------------------------------------")
        print("Current epoch: ", currentMetrics['epoch'])
        print("Precision: ", currentMetrics['precision'])
        print("Recall: ", currentMetrics['recall'])
        print("F1: ", currentMetrics['F1'])
        print("NDCG: ", currentMetrics['NDCG'])
        print("------------------------------------------------")
        print("Best epoch: ", self.bestMetrics['epoch'])
        print("Precision: ", self.bestMetrics['precision'])
        print("Recall: ", self.bestMetrics['recall'])
        print("F1: ", self.bestMetrics['F1'])
        print("NDCG: ", self.bestMetrics['NDCG'])
        print("------------------------------------------------")
        
    def precision(self):
        return sum(self.hits.values()) / len(self.hits) / self.top_k
    
    def recall(self):
        return sum([self.hits[user] / len(self.prompt_profile[user]) for user in np.unique(self.prompt_set['users'])]) / len(self.hits)
        
    def F1(self, precision, recall):
        if precision + recall != 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0.
      
    def NDCG(self):
        NDCG = 0
        for i, user in enumerate(np.unique(self.prompt_set['users'])):
            DCG = 0
            IDCG = 0
            for n, item in enumerate(self.rec_list[user]):
                if item in self.prompt_profile[user]:
                    DCG += 1.0 / np.log(n+2)
                IDCG += 1.0 / np.log(n+2)
            NDCG += DCG / IDCG
        return NDCG / (i+1)
    
    #optimized 
    def find_k_largest(self, K, candidates):
        n_candidates = []
        
        #initialize-random get K items
        for iid,score in enumerate(candidates[:K]):
            n_candidates.append((iid, score))
        n_candidates.sort(key=lambda d: d[1], reverse=True) #sort by score from large to small
        k_largest_scores = [item[1] for item in n_candidates]
        ids = [item[0] for item in n_candidates]
        
        #find the N biggest scores
        for iid,score in enumerate(candidates):
            ind = K #final order
            l = 0 #head
            r = K - 1 #tail
            if k_largest_scores[r] < score:
                #bisection method
                while r >= l:
                    mid = int((r - l) / 2) + l
                    if k_largest_scores[mid] >= score:
                        l = mid + 1
                    elif k_largest_scores[mid] < score:
                        r = mid - 1
                    if r < l:
                        ind = r
                        break
            # move the items backwards
            if ind < K - 2:
                k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
                ids[ind + 2:] = ids[ind + 1:-1]
            if ind < K - 1:
                k_largest_scores[ind + 1] = score
                ids[ind + 1] = iid
        return ids,k_largest_scores
    
class LightGCN(data, evaluation):
    def __init__(self):
        super(LightGCN, self).__init__()
        
        #general parameter
        self.emb_size = 50
        self.reg = 0.001
        self.learning_rate = 0.001 
        self.maxEpoch = 1
        self.batch_size = 2000
        self.top_k = 10
        self.earlystop = 10
        self.bestMetrics = {'epoch':0, 'precision':0, 'recall':0, 'F1':0, 'NDCG':0}
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='item_embeddings')
        
        #model specific parameter
        self.n_layers = 2
        
        #run model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model()
        self.trainModel()
        
    def sparse2Tensor(self, sp_adj):
        row, col = sp_adj.nonzero()
        indices = np.array(list(zip(row, col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=sp_adj.data, dense_shape=sp_adj.shape)
        return adj_tensor 
    
    def model(self):
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        norm_adj = self.sparse2Tensor(self.joint_R)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.reduce_mean(all_embeddings, axis=0)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], axis=0)
        
        #get the embeddings
        self.u_idx = tf.placeholder(tf.int32, shape=[None])
        self.i_idx = tf.placeholder(tf.int32, shape=[None])
        self.neg_idx = tf.placeholder(tf.int32, shape=[None])
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.i_idx)
        
    def trainModel(self):
        rec_loss = self.bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.reg * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(self.batch_neg_item_emb))
        opt = tf.train.AdamOptimizer(self.learning_rate)
        train = opt.minimize(rec_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        variable_names = [v.name for v in tf.trainable_variables()]
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.onePosOneNegBatch()):
                u, i, neg = batch
                #print(u_idx,i_idx)
                _, l = self.sess.run([train, rec_loss], feed_dict={self.u_idx: u, self.i_idx: i, self.neg_idx: neg})
                print("Training:", epoch+1, "Batch:", n+1, "Loss:", l)
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])
            self.predictForRanking(prompt_name='test', epoch=epoch+1)
            #early stop
            if epoch - self.bestMetrics['epoch'] == self.earlystop:
                print("Early stop!!!")
                break
        #self.predictForRanking(name='test')
        
        #output logs
        filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S ") + "LightGCN.txt"
        with open(os.path.join('logs',filename), 'w') as fp:
            fp.write("-----------------------------------------------------\n")
            fp.write("PARAMETER SETTING\n")
            fp.write("Embedding size: "+str(self.emb_size)+"\n")
            fp.write("L2-Regularization:"+str(self.reg)+"\n")
            fp.write("Learning rate: "+str(self.learning_rate)+"\n")
            fp.write("Max epoch: "+str(self.maxEpoch)+"\n")
            fp.write("Batch size: "+str(self.batch_size)+"\n")
            fp.write("Top K value: "+str(self.top_k)+"\n")
            fp.write("Early stop: "+str(self.earlystop)+"\n")
            fp.write("Graph layer: "+str(self.n_layers)+"\n")
            fp.write("-----------------------------------------------------\n")
            
            if epoch+1 != self.maxEpoch:
                fp.write("Early stop at %d/%d.\n" %(epoch+1, self.maxEpoch))
            for key in self.bestMetrics.keys():
                fp.write(str(key)+': '+str(self.bestMetrics[key])+'\n')    

    def onePosOneNegBatch(self):
        train_size = len(self.train['users'])
        r_index = np.arange(train_size)
        np.random.shuffle(r_index)
        training_user = self.train['users'][r_index]
        training_item = self.train['pos_item'][r_index]
        batch_id = 0
        item_cache = np.array(list(self.item2id.keys()))
        
        #split batch
        while batch_id < train_size:
            if batch_id + self.batch_size <= train_size:
                users = training_user[batch_id:batch_id+self.batch_size]
                items = training_item[batch_id:batch_id+self.batch_size]
                batch_id += self.batch_size
            else:
                users = training_user[batch_id:]
                items = training_item[batch_id:]
                batch_id = train_size
            
            #user2id,item2id
            u_idx, i_idx, neg_idx = [], [], []
            for pair in zip(users, items):
                u_idx.append(self.user2id[pair[0]])
                i_idx.append(self.item2id[pair[1]])
                
                neg_item = np.random.choice(item_cache)
                while neg_item in self.user_train_profile[pair[0]]:
                    neg_item = np.random.choice(item_cache)
                neg_idx.append(self.item2id[neg_item])
            yield u_idx, i_idx, neg_idx

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        score = tf.reduce_sum(tf.multiply(user_emb, pos_emb), axis=1) - tf.reduce_sum(tf.multiply(user_emb, neg_emb), axis=1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(score)+10e-8))
        return loss
        
new = LightGCN()
    