# -*- coding: utf-8 -*-
"""Create the Data Set for the Experiments.
===============================================
Version |    Date     |   Author    |   Comment
-----------------------------------------------
0.0     | 25 Oct 2020 | J. Charlier | initial version
===============================================
Models: {
    FNN 3 layers, FNN 5 layers, FNN 10 layers,
    CNN 3 layers, CNN 5 layers, CNN 10 layers,
    CNN Replica Lin et al., RF
}
Data:
*   encoded4x23 pkl file: preprocessed 4x23 encoded crispor data
*   encoded8x23 pkl file: preprocessed 8x23 encoded crispor data
*   encoded4x23linn pkl file: preprocessed 4x23 encoded crispor data with data 
        provided in Linn et al.
*   encoded8x23linn pkl file: preprocessed 8x23 encoded crispor data with data 
        provided in Linn et al.
*   encoded4x23withouttsai pkl file: preprocessed 4x23 encoded crispor data
        without tsai guide seq data
*   encoded8x23withouttsai pkl file: preprocessed 8x23 encoded crispor data
        without tsai guide seq data
Useful Link
  Neural Nets Hyperparameters Tuning
  https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
"""
#
#
from __future__ import print_function
import os
import time
import random
random.seed(42)
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score,
                             roc_curve, precision_score, recall_score,
                             auc, average_precision_score, 
                             precision_recall_curve, accuracy_score)
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow.python.keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (Conv2D, MaxPooling2D, MaxPool2D,
                                            concatenate, BatchNormalization, 
                                            Dense, Dropout, Flatten, Input)
from tensorflow.python.keras.preprocessing.image import (ImageDataGenerator,
                                       img_to_array, 
                                       array_to_img)
import tensorflow.python.keras as tfkeras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
#
import utilities
import ffns
import cnns
import mltrees
p = print
#
#
# Incorporating reduced learning and early stopping for NN callback
reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, 
    patience=8, verbose=1, 
    mode='auto', min_delta=0.02, 
    cooldown=0, min_lr=0)
eary_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    patience=20, verbose=1, mode='auto')
callbacks = [reduce_learning, eary_stopping]
#
#
ismodelsaved = True   
#
#
# data read
# -*-*-*-*-
imgrows = 4
nexp = 3
imgcols = 23
num_classes = 2
epochs = 500
batch_size = 64
undersampling = False
#
# we import the pkl file containing the data
flpath = ''
loaddata = utilities.importData(
    flpath=flpath,
    encoding='4x23',
    sim='crispor',
    tl=False 
)
#
# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    pd.Series(loaddata.target),
    test_size=0.3,
    shuffle=True, 
    random_state=42
)
#
#
p('\n!!! train ffns !!!\n')
xtrainffn, xtestffn, ytrainfnn, ytestffn, inputshapeffn = ffns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes
)
ffn3 = ffns.ffnthree(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
ffn5 = ffns.ffnfive(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
ffn10 = ffns.ffnten(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
#
p('\n!!! train cnns !!!\n')
xtraincnn, xtestcnn, ytraincnn, ytestcnn, inputshapecnn = cnns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes
)
cnn3 = cnns.cnnthree(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
cnn5 = cnns.cnnfive(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
cnn10 = cnns.cnnten(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
)
cnnlin = cnns.cnnlin(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved
) 
#
p('\n!!! train random forest !!!\n')
xtrainrf, xtestrf, ytrainrf, ytestrf = mltrees.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols
)
rf = mltrees.initFitRF(xtrainrf, ytrainrf)
#
#
p('\n!!!! roc curve on crispor data !!!\n')
utilities.plotRocCurve(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10, cnnlin, rf
    ],
    [
        'FFN3', 'FFN5',
        'FFN10', 'CNN3',
        'CNN5', 'CNN10', 'CNN Lin', 'RF'
    ],
    [
        xtestffn, xtestffn,
        xtestffn, xtestcnn,
        xtestcnn, xtestcnn, xtestcnn, xtestrf
    ],
    [
        ytestffn, ytestffn,
        ytestffn, ytestcnn,
        ytestcnn, ytestcnn, ytestcnn, y_test
    ],
    'roccurvecrispr4x23.pdf'
)
p('\n!!!! precision recall curve on crispor data !!!\n')
utilities.plotPrecisionRecallCurve(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10, cnnlin, rf
    ],
    [
        'FFN3', 'FFN5',
        'FFN10', 'CNN3',
        'CNN5', 'CNN10',
        'CNN Lin', 'RF'
    ],
    [
        xtestffn, xtestffn,
        xtestffn, xtestcnn,
        xtestcnn, xtestcnn,
        xtestcnn, xtestrf
    ],
    [
        ytestffn, ytestffn,
        ytestffn, ytestcnn,
        ytestcnn, ytestcnn,
        ytestcnn, y_test
    ],
    'precisionrecallcurvecrispr4x23.pdf'
)
#
#
preds = utilities.collectPreds(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10,
        cnnlin, rf
    ],
    [
        xtestffn, xtestffn,
        xtestffn, xtestcnn,
        xtestcnn, xtestcnn,
        xtestcnn, xtestrf
    ]
)
# correct predictions of Linn et al.
preds.yscore[-2][:, 1] = np.abs(preds.yscore[-2][:, 1])
for n in range(len(preds.yscore[-2])):
    under = preds.yscore[-2][n, 0] + preds.yscore[-2][n, 1]
    preds.yscore[-2][n, 0] = preds.yscore[-2][n, 0] / (under)
    preds.yscore[-2][n, 1] = preds.yscore[-2][n, 1] / (under)
#
for objfun in [utilities.brierScore, accuracy_score, f1_score, precision_score, recall_score]:
    if 'brier' in str(objfun):
        utilities.computeScore(objfun, y_test, preds.yscore)
    else:
        utilities.computeScore(objfun, y_test, preds.ypred)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    y_test,
    loaddata.target_names,
    imgrows
)
#
issave = False
if issave:
    ffnthree.save('saved_model/ffn3_4x23')
    ffnfive.save('saved_model/ffn5_4x23')
    ffnten.save('saved_model/ffn10_4x23')
    cnnthree.save('saved_model/cnn3_4x23')
    cnnfive.save('saved_model/cnn5_4x23')
    cnnten.save('saved_model/cnn10_4x23')
    cnnlin.save('saved_model/cnnlinn_4x23')
#
#
p('\n\n')
p('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
p('!!!       PREDICTIONS ON GUIDE SEQ           !!!')
p('!!!       RESULTS FOR PUBLICATION            !!!')
p('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
loadguideseq = utilities.importData(
    flpath=flpath,
    encoding='4x23',
    sim='guideseq',
    tl=False
)
gseq = utilities.transformGuideSeqImages(
		loadguideseq,
		num_classes,
		imgrows, imgcols
)
p('\n!!!! roc curve on guideseq data !!!\n')
utilities.plotRocCurve(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10, cnnlin, rf
    ],
    [
        'FFN3', 'FFN5',
        'FFN10', 'CNN3',
        'CNN5', 'CNN10', 'CNN Lin', 'RF'
    ],
    [
        gseq.xgseqffn, gseq.xgseqffn,
        gseq.xgseqffn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqrf
    ],
    [
        gseq.ygseqffn, gseq.ygseqffn,
        gseq.ygseqffn, gseq.ygseqcnn,
        gseq.ygseqcnn, gseq.ygseqcnn,
        gseq.ygseqcnn, gseq.ygseqrf
    ],
    'roccurveguideseq4x23.pdf'
)
p('\n!!!! precision recall curve on guideseq data !!!\n')
utilities.plotPrecisionRecallCurve(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10, cnnlin, rf
    ],
    [
        'FFN3', 'FFN5',
        'FFN10', 'CNN3',
        'CNN5', 'CNN10', 'CNN Lin', 'RF'
    ],
    [
        gseq.xgseqffn, gseq.xgseqffn,
        gseq.xgseqffn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqrf
    ],
    [
        gseq.ygseqffn, gseq.ygseqffn,
        gseq.ygseqffn, gseq.ygseqcnn,
        gseq.ygseqcnn, gseq.ygseqcnn,
        gseq.ygseqcnn, gseq.ygseqrf
    ],
    'precisionrecallcurveguideseq4x23.pdf'
)
predsgseq = utilities.collectPreds(
    [
        ffn3, ffn5,
        ffn10, cnn3,
        cnn5, cnn10,
        cnnlin, rf
    ],
    [
        gseq.xgseqffn, gseq.xgseqffn,
        gseq.xgseqffn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqcnn,
        gseq.xgseqcnn, gseq.xgseqrf
    ]
)
# correct predictions of Linn et al.
predsgseq.yscore[-2][:, 1] = np.abs(predsgseq.yscore[-2][:, 1])
for n in range(len(predsgseq.yscore[-2])):
    under = predsgseq.yscore[-2][n, 0] + predsgseq.yscore[-2][n, 1]
    predsgseq.yscore[-2][n, 0] = predsgseq.yscore[-2][n, 0] / (under)
    predsgseq.yscore[-2][n, 1] = predsgseq.yscore[-2][n, 1] / (under)
#
for objfun in [utilities.brierScore, accuracy_score, f1_score, precision_score, recall_score]:
    if 'brier' in str(objfun):
        utilities.computeScore(objfun, loadguideseq.target, predsgseq.yscore)
    else:
        utilities.computeScore(objfun, loadguideseq.target, predsgseq.ypred)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    gseq.ygseqdf,
    loadguideseq.target_names,
    imgrows
)
#
# Last card of module offtargetmodelsexperiments4x23.
#