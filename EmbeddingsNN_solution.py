
# Pytorch Embedding NN code:
# https://www.kaggle.com/sapthrishi007/jane-fastai-embedding-smoothnn5-300x5/notebook?scriptVersionId=53603141&select=Jane_EmbNN5_auc_400_400_400.pth

# Apart from the 130 featres, it also creates feature embeddings from their tags in features.csv file.

# pytorch train code:
# https://www.kaggle.com/a763337092/neural-network-starter-pytorch-version https://www.kaggle.com/a763337092/pytorch-resnet-starter-training

# tensorflow training code:
# https://www.kaggle.com/code1110/jane-street-with-keras-nn-overfit

# blending pytorch and tensorflow code:
# https://www.kaggle.com/a763337092/blending-tensorflow-and-pytorch

# Fork from:
# https://www.kaggle.com/sapthrishi007/pytorch-embeddingsnn-resnet-tensorflow

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# /kaggle/input/jane-street-market-prediction/example_sample_submission.csv
# /kaggle/input/jane-street-market-prediction/features.csv
# /kaggle/input/jane-street-market-prediction/example_test.csv
# /kaggle/input/jane-street-market-prediction/train.csv
# /kaggle/input/jane-street-market-prediction/janestreet/competition.cpython-37m-x86_64-linux-gnu.so
# /kaggle/input/jane-street-market-prediction/janestreet/__init__.py
# /kaggle/input/jane-street-with-keras-nn-overfit/__results__.html
# /kaggle/input/jane-street-with-keras-nn-overfit/submission.csv
# /kaggle/input/jane-street-with-keras-nn-overfit/__resultx__.html
# /kaggle/input/jane-street-with-keras-nn-overfit/__notebook__.ipynb
# /kaggle/input/jane-street-with-keras-nn-overfit/model.h5
# /kaggle/input/jane-street-with-keras-nn-overfit/__output__.json
# /kaggle/input/jane-street-with-keras-nn-overfit/custom.css
# /kaggle/input/mlp012003weights/online_model4.pth
# /kaggle/input/mlp012003weights/online_model2.pth
# /kaggle/input/mlp012003weights/online_model1.pth
# /kaggle/input/mlp012003weights/f_mean_online.npy
# /kaggle/input/mlp012003weights/online_model3.pth
# /kaggle/input/mlp012003weights/online_model0.pth
# /kaggle/input/janefastai-featinteraction-embeddingnn5-300x3/Jane_EmbNN5_auc_400_400_400.pth
# /kaggle/input/janefastai-featinteraction-embeddingnn5-300x3/submission.csv


# pytorch Resnet part
import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import warnings
warnings.filterwarnings ("ignore")

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = '../input/jane-street-market-prediction/'

NFOLDS = 5

TRAIN = False
CACHE_PATH = '../input/mlp012003weights'

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
    # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
    # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

feat_cols = [f'feature_{i}' for i in range(130)]

target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']

f_mean = np.load(f'{CACHE_PATH}/f_mean_online.npy')

##### Making features
all_feat_cols = [col for col in feat_cols]
all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

##### Model&Data fnc
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        dropout_rate = 0.2
        hidden_size = 256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x

if True:
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    model_list = []
    tmp = np.zeros(len(feat_cols))
    for _fold in range(NFOLDS):
        torch.cuda.empty_cache()
        model = Model()
        model.to(device)
        model_weights = f"{CACHE_PATH}/online_model{_fold}.pth"
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
        model.eval()
        model_list.append(model)
      
# tensorflow part
!ls ../input/jane-street-with-keras-nn-overfit
# __notebook__.ipynb  __results__.html  custom.css  submission.csv
# __output__.json     __resultx__.html  model.h5

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choices


SEED = 1111

np.random.seed(SEED)

# fit
def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)
    
    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model

epochs = 200
batch_size = 4096
hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
clf = create_mlp(
    len(feat_cols), 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )
clf.load_weights('../input/jane-street-with-keras-nn-overfit/model.h5')

tf_models = [clf]
EMbeddings NN
N_FEAT_TAGS = 29    # No of tags in features.csv
DEVICE = device
N_FEATURES = 130
THREE_HIDDEN_LAYERS = [400, 400, 400]

class FFN (nn.Module):
    
    def __init__(self, inputCount=130, outputCount=5, hiddenLayerCounts=[150, 150, 150], 
                 drop_prob=0.2, nonlin=nn.SiLU (), isOpAct=False):
        
        super(FFN, self).__init__()
        
        self.nonlin     = nonlin
        self.dropout    = nn.Dropout (drop_prob)
        self.batchnorm0 = nn.BatchNorm1d (inputCount)
        self.dense1     = nn.Linear (inputCount, hiddenLayerCounts[0])
        self.batchnorm1 = nn.BatchNorm1d (hiddenLayerCounts[0])
        self.dense2     = nn.Linear(hiddenLayerCounts[0], hiddenLayerCounts[1])
        self.batchnorm2 = nn.BatchNorm1d (hiddenLayerCounts[1])
        self.dense3     = nn.Linear(hiddenLayerCounts[1], hiddenLayerCounts[2])
        self.batchnorm3 = nn.BatchNorm1d (hiddenLayerCounts[2])        
        self.outDense   = None
        if outputCount > 0:
            self.outDense   = nn.Linear (hiddenLayerCounts[-1], outputCount)
        self.outActivtn = None
        if isOpAct:
            if outputCount == 1 or outputCount == 2:
                self.outActivtn = nn.Sigmoid ()
            elif outputCount > 0:
                self.outActivtn = nn.Softmax (dim=-1)
        return

    def forward (self, X):
        
        # X = self.dropout (self.batchnorm0 (X))
        X = self.batchnorm0 (X)
        X = self.dropout (self.nonlin (self.batchnorm1 (self.dense1 (X))))
        X = self.dropout (self.nonlin (self.batchnorm2 (self.dense2 (X))))
        X = self.dropout (self.nonlin (self.batchnorm3 (self.dense3 (X))))
        if self.outDense:
            X = self.outDense (X)
        if self.outActivtn:
            X = self.outActivtn (X)
        return X
    
    
class Emb_NN_Model (nn.Module):
    
    def __init__(self, three_hidden_layers=THREE_HIDDEN_LAYERS, embed_dim=(N_FEAT_TAGS), csv_file='../input/jane-street-market-prediction/features.csv'):
        
        super (Emb_NN_Model, self).__init__()
        global N_FEAT_TAGS
        N_FEAT_TAGS = 29
        
        # store the features to tags mapping as a datframe tdf, feature_i mapping is in tdf[i, :]
        dtype = {'tag_0' : 'int8'}
        for i in range (1, 29):
            k = 'tag_' + str (i)
            dtype[k] = 'int8'
        t_df = pd.read_csv (csv_file, usecols=range (1,N_FEAT_TAGS+1), dtype=dtype)
        t_df['tag_29'] = np.array ([1] + ([0] * (t_df.shape[0]-1)) ).astype ('int8')
        self.features_tag_matrix = torch.tensor (t_df.to_numpy ())
        N_FEAT_TAGS += 1
        
        # print ('self.features_tag_matrix =', self.features_tag_matrix)
        
        # embeddings for the tags. Each feature is taken a an embedding which is an avg. of its' tag embeddings
        self.embed_dim     = embed_dim
        self.tag_embedding = nn.Embedding (N_FEAT_TAGS+1, embed_dim) # create a special tag if not known tag for any feature
        self.tag_weights   = nn.Linear (N_FEAT_TAGS, 1)
        
        drop_prob          = 0.5
        self.ffn           = FFN (inputCount=(130+embed_dim), outputCount=0, hiddenLayerCounts=[(three_hidden_layers[0]+embed_dim), (three_hidden_layers[1]+embed_dim), (three_hidden_layers[2]+embed_dim)], drop_prob=drop_prob)
        self.outDense      = nn.Linear (three_hidden_layers[2]+embed_dim, 5)
        return
    
    def features2emb (self):
        """
        idx : int feature index 0 to N_FEATURES-1 (129)
        """
        
        all_tag_idxs = torch.LongTensor (np.arange (N_FEAT_TAGS)) #.to (DEVICE)              # (29,)
        tag_bools    = self.features_tag_matrix                                # (130, 29)
        # print ('tag_bools.shape =', tag_bools.size())
        f_emb        = self.tag_embedding (all_tag_idxs).repeat (130, 1, 1)    #;print ('1. f_emb =', f_emb) # (29, 7) * (130, 1, 1) = (130, 29, 7)
        # print ('f_emb.shape =', f_emb.size())
        f_emb        = f_emb * tag_bools[:, :, None]                           #;print ('2. f_emb =', f_emb) # (130, 29, 7) * (130, 29, 1) = (130, 29, 7)
        # print ('f_emb.shape =', f_emb.size())
        
        # Take avg. of all the present tag's embeddings to get the embedding for a feature
        s = torch.sum (tag_bools, dim=1)                                       # (130,)
        # print ('s =', s)              
        f_emb = torch.sum (f_emb, dim=-2) / s[:, None]                         # (130, 7)
        # print ('f_emb =', f_emb)        
        # print ('f_emb.shape =', f_emb.shape)
        
        # take a linear combination of the present tag's embeddings
        # f_emb = f_emb.permute (0, 2, 1)                                        # (130, 7, 29)
        # f_emb = self.tag_weights (f_emb)                      #;print ('3. f_emb =', f_emb)                 # (130, 7, 1)
        # f_emb = torch.squeeze (f_emb, dim=-1)                 #;print ('4. f_emb =', f_emb)                 # (130, 7)
        return f_emb
    
    def forward (self, cat_featrs, features):
        """
        when you call `model (x ,y, z, ...)` then this method is invoked
        """
        
        cat_featrs = None
        features   = features.view (-1, N_FEATURES)
        f_emb      = self.features2emb ()                                #;print ('5. f_emb =', f_emb); print ('6. features =', features) # (130, 7)
        # print ('features.shape =', features.shape, 'f_emb.shape =', f_emb.shape)
        features_2 = torch.matmul (features, f_emb)                      #;print ('7. features =', features) # (1, 130) * (130, 7) = (1, 7)
        # print ('features.shape =', features.shape)
        
        # Concatenate the two features (features + their embeddings)
        features   = torch.hstack ((features, features_2))        
        
        x          = self.ffn (features)                               #;print ('8. x.shape = ', x.shape, 'x =', x)   # (1, 7) -> (1, 7)
        # x        = self.layer_normal (x + features)                  #;print ('9. x.shape = ', x.shape, 'x =', x)   # (1, 7) -> (1, 2)
        out_logits = self.outDense (x)                                 #;print ('10. out_logits.shape = ', out_logits.shape, 'out_logits =', out_logits)        
        # return sigmoid probs
        # out_probs = F.sigmoid (out_logits)
        return out_logits
embNN_model = Emb_NN_Model ()

try:
    embNN_model.load_state_dict (torch.load ("../input/janefastai-featinteraction-embeddingnn5-300x3/Jane_EmbNN5_auc_400_400_400.pth"))
except:
    embNN_model.load_state_dict (torch.load ("../input/janefastai-featinteraction-embeddingnn5-300x3/Jane_EmbNN5_auc_400_400_400.pth", map_location='cpu'))
    
embNN_model = embNN_model.eval ()
# Inference
import janestreet
env = janestreet.make_env()
env_iter = env.iter_test()
if True:

    for (test_df, pred_df) in tqdm(env_iter):
        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, feat_cols].values
            if np.isnan(x_tt.sum()):
                x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * f_mean

            cross_41_42_43 = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
            cross_1_2 = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
            feature_inp = np.concatenate((
                x_tt,
                np.array(cross_41_42_43).reshape(x_tt.shape[0], 1),
                np.array(cross_1_2).reshape(x_tt.shape[0], 1),
            ), axis=1)

            # torch_pred
            torch_pred = np.zeros((1, len(target_cols)))
            for model in model_list:
                torch_pred += model(torch.tensor(feature_inp, dtype=torch.float).to(device)).sigmoid().detach().cpu().numpy() / NFOLDS
            torch_pred = np.median(torch_pred)
            
            # tf_pred
            tf_pred = np.median(np.mean([model(x_tt, training = False).numpy() for model in tf_models],axis=0))
            
            # torch embedding_NN pred
            x_tt    = torch.tensor (x_tt).float ().view (-1, 130)
            embnn_p = np.median (torch.sigmoid (embNN_model (None, x_tt)).detach ().cpu ().numpy ().reshape ((-1, 5)), axis=1)   # not logits, actually sigmoid probabilities
            
            # avg
            pred_pr = torch_pred*0.42 + tf_pred*0.42 + embnn_p*0.16
            
            pred_df.action = np.where (pred_pr >= 0.4978, 1, 0).astype (int)
        else:
            pred_df.action = 0
        env.predict(pred_df)

 # https://www.kaggle.com/code/sagarjiyani/pytorch-embeddingsnn-resnet
