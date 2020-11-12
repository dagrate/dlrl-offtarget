"""Machine Learning Tree-Based Methods.
===============================================
Version |Date |   Author|   Comment
-----------------------------------------------
0.0 | 31 Oct 2020 | J. Charlier | initial version
===============================================
"""
#
#
from sklearn.ensemble import RandomForestClassifier
p = print
#
#
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols):
    xtrain = xtrain.astype('float32').reshape(-1, imgrows*imgcols)
    xtest = xtest.astype('float32').reshape(-1, imgrows*imgcols)
    xtrain /= 255
    xtest /= 255
    p('xtrain shape:', xtrain.shape)
    p(xtrain.shape[0], 'train samples')
    p(xtest.shape[0], 'test samples')
    return xtrain, xtest, ytrain, ytest
#
#
def initFitRF(xtrain, ytrain):
    rf = RandomForestClassifier(
        bootstrap=True, ccp_alpha=0.0, class_weight=None,
        criterion='gini', max_depth=None, max_features='auto',
        max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=1000,
        n_jobs=None, oob_score=False, random_state=42,
        verbose=0, warm_start=False
        )
    rf.fit(xtrain, ytrain)
    p("RF Training: Done")
    return rf
#
# Last card of module mltrees.
#
