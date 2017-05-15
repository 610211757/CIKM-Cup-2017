from sklearn import ensemble
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import decomposition
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.externals import joblib

def getDataSet1(n):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    dataSet_label = []
    i = 0
    k = 1000 * n
    f = open('E:\\李卓聪\\rain_predict\\data_new\\CIKM2017_train\\train.txt','r')
    for line in f.readlines():
        if(i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            if(rainfall <= 5 and rainfall>0 ):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(1)
            if(rainfall>5 and rainfall<=19 ):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(2)
            if(rainfall>19 and rainfall<=42 ):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(3)
            if(rainfall>42 ):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(4)
            print(i)
            if(i == k+ 999):
                break
        i += 1
    return testName_all, rainfall_all, dataSet_all, dataSet_label


def getDataSet2(n, label_num):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    dataSet_label = []
    i = 0
    k = 1000 * n
    f = open('E:\\李卓聪\\rain_predict\\data_new\\CIKM2017_train\\train.txt','r')
    for line in f.readlines():
        if(i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            if(rainfall <= 5 and rainfall>0 and label_num==1):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(1)
            if(rainfall>5 and rainfall<=19 and label_num==2):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(2)
            if(rainfall>19 and rainfall<=42 and label_num==3):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(3)
            if(rainfall>42 and label_num==4):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(4)
            print(i)
            if(i == k+ 999):
                break
        i += 1
    return testName_all, rainfall_all, dataSet_all, dataSet_label


def pre_train():
    dataSet_all = []
    rainfall_all = []
    dataSet_label_all= []
    print("aaaaaaaaaaaaaaaaa")
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    print("bbbbbbbbbbbbbbbb")
    for i in range(10):
        testName, rainfall, dataSet, dataSet_label = getDataSet1(i)
        dataSet = ipca.fit_transform(dataSet)
        dataSet_all.extend(dataSet)
        rainfall_all.extend(rainfall)
        dataSet_label_all.extend(dataSet_label)

    gbc = GradientBoostingClassifier()
    gbc.fit(dataSet_all, dataSet_label_all)
    joblib.dump(gbc, "rain_pretrain_GBDR.m")
    print("ddddddddddddddd")

def train():
    dataSet_all = [[],[],[],[],[]]
    rainfall_all = [[],[],[],[],[]]
    dataSet_label_all = [[],[],[],[],[]]
    print("aaaaaaaaaaaaaaaaa")
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    print("bbbbbbbbbbbbbbbb")
    for k in range(4):
        for i in range(10):
            testName, rainfall, dataSet, dataSet_label = getDataSet2(i, k)
            dataSet = ipca.fit_transform(dataSet)
            dataSet_all[k].extend(dataSet)
            rainfall_all[k].extend(rainfall)
            dataSet_label_all[k].extend(dataSet_label)

    for i in range(4):
        gbdt = ensemble.GradientBoostingRegressor()
        gbdt.fit(dataSet_all[i], rainfall_all[i])  # 训练数据来学习，不需要返回值
        os.chdir("E:\\李卓聪\\save_file\\machine_model")
        joblib.dump(gbdt, "rain_train_GBDT_"+str(i)+".m")
        print("ddddddddddddddd")


def predict():
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    gbdt1 = joblib.load("rain_train_GBDT_1.m")
    gbdt2 = joblib.load("rain_train_GBDT_2.m")
    gbdt3 = joblib.load("rain_train_GBDT_3.m")
    gbdt4 = joblib.load("rain_train_GBDT_4.m")
    result = []
    for i in range(2):
        print("1")
        testName, rainfall, dataSet, dataSet_label = getDataSet1(i)
        print("2")
        dataSet = ipca.fit_transform(dataSet)
        print("3")
        for k in range(1000):
            if(dataSet_label[k] == 1):
                y = gbdt1.predict(dataSet[k])
            if (dataSet_label[k] == 2):
                y = gbdt2.predict(dataSet[k])
            if (dataSet_label[k] == 3):
                y = gbdt3.predict(dataSet[k])
            if (dataSet_label[k] == 4):
                y = gbdt4.predict(dataSet[k])
            result.append(y)

        print("4")
        save_file = "E:\\李卓聪\\save_file\\rainfall_predict\\" + "GBDT_result_" + str(i) + ".txt"
        fl = open(save_file, 'w')
        for k in y:
            fl.write(str(k))
            fl.write("\n")
        fl.close()
        print("5")
    print("6")




if __name__ == "__main__":
    pre_train()
    print("pretrain done!")
    train()
    print("train done!")
    predict()
    print("finish!")























