# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
import os
import datetime
import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import sys
from src.preprocess import feature_eng
import config
# %%
# load the data
data = np.load(
    'simulated zero/existance npz file path')
# data = np.load('result/simu_reading_2mag/2021-02-02 22:21_100000.0.npz')
X = data['reading']
# X = feature_eng(X, config.pSensor_small_smt)
Y = data['label']
print(X.shape)
# Y[Y == 2] = 1

# %%
# keep type 1 and 2
idx = Y > 0
X = X[idx]
Y = Y[idx]-1
# %%
# SVM算法

starttime = datetime.datetime.now()
run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)


# #自动调参函数
if not os.path.exists('result/svm_log'):
    os.makedirs('result/svm_log')
if not os.path.exists('result/svm'):
    os.makedirs('result/svm')
with open('result/svm_log/{}.txt'.format(run_time), 'w') as f:
    sys.stdout = f
    tuned_parameters = [
        {'kernel': ['rbf'],
         'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-9, 1e-10, 1e-12],
         'C': [10, 100, 1000, 2000, 3000]}]
    # scores = ['precision_macro', 'recall_macro', 'roc_auc']
    scores = ['f1_macro', 'roc_auc']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        # cv为迭代次数。#基于交叉验证的网格搜索，cv:确定交叉验证拆分策略。
        clf = ms.GridSearchCV(svm.SVC(), tuned_parameters,
                              cv=5, scoring=score, n_jobs=-1)
        # 用训练集训练这个学习器 clf
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        # save the classifier

        with open('result/svm/{}_{}.pkl'.format(run_time, score), 'wb') as fid:
            pickle.dump(clf, fid)

        # load it again
        with open('result/svm/{}_{}.pkl'.format(run_time, score), 'rb') as fid:
            new_clf = pickle.load(fid)
        y_true, y_pred = y_test, new_clf.predict(X_test)

        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))

        print(confusion_matrix(y_true, y_pred))
