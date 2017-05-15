from sklearn import ensemble
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import decomposition
import os
from sklearn.externals import joblib


def getDataSet1(n):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
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
            testName_all.append(testName)
            rainfall_all.append(rainfall)
            dataSet_all.append(dataSet)
            print(i)
            if(i == k+ 999):
                break
        i += 1
    return testName_all,  rainfall_all, dataSet_all


def getDataSet2(n):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    i = 0
    k = 1000 * n
    f = open('E:\\李卓聪\\rain_predict\\data_new\\CIKM2017_testA\\testA.txt','r')
    for line in f.readlines():
        if(i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            testName_all.append(testName)
            rainfall_all.append(rainfall)
            dataSet_all.append(dataSet)
            print(i)
            if(i == k+ 999):
                break
        i += 1
    return testName_all,  rainfall_all, dataSet_all


def get_pca():
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = IncrementalPCA(n_components=1000)
    # ipca = joblib.load("rain_ipca_model.m")
    for i in range(10):
        testName, rainfall, dataSet = getDataSet1(i)
        print("done get data")
        ipca.partial_fit(dataSet)
        print("fit done =",i)
    joblib.dump(ipca, "rain_ipca_model.m")


def train():
    dataSet_all = []
    rainfall_all = []
    print("aaaaaaaaaaaaaaaaa")
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    print("bbbbbbbbbbbbbbbb")
    for i in range(10):
        testName, rainfall, dataSet = getDataSet1(i)
        dataSet = ipca.fit_transform(dataSet)
        dataSet_all.extend(dataSet)
        rainfall_all.extend(rainfall)

    print("cccccccccccccccc")
    gbdt = ensemble.GradientBoostingRegressor()
    gbdt.fit(dataSet_all, rainfall_all)  # 训练数据来学习，不需要返回值
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    joblib.dump(gbdt, "rain_train_GBDT.m")
    print("ddddddddddddddd")

    # for i in range(len(dataSet)):
    #     save_file = "E:\\李卓聪\\save_file\\rain\\" + str(i) + ".txt"
    #     fl = open(save_file, 'w')
    #     for k in dataSet[i]:
    #         fl.write(str(k))
    #         fl.write("\n")
    #     fl.close()


def predict():
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    gbdt = joblib.load("rain_train_GBDT.m")
    for i in range(2):
        print("1")
        testName, rainfall, dataSet = getDataSet2(i)
        print("2")
        dataSet = ipca.fit_transform(dataSet)
        print("3")
        y = gbdt.predict(dataSet)
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
    get_pca()
    print("get_pca done!")
    train()
    print("train done!")
    predict()
    print("finish!")
















