from sklearn import ensemble
import numpy as np
from sklearn import decomposition
import os
from sklearn.externals import joblib


def getDataSet1(n, fileName):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    i = 0
    k = 1000 * n
    if fileName == 'train':
        f = open('E:\\李卓聪\\rain_predict\\data_new\\CIKM2017_train\\train.txt','r')
    if fileName == 'test':
        f = open('E:\\李卓聪\\rain_predict\\data_new\\CIKM2017_testA\\testA.txt', 'r')
    for line in f.readlines():
        if(i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            dataSet = dataConvert(dataSet)
            rainfall = float(rainfall)
            testName_all.append(testName)
            rainfall_all.append(rainfall)
            dataSet_all.append(dataSet)
            print(i)
            if(i == k+ 999):
                break
        i += 1
    return testName_all,  rainfall_all, dataSet_all


def dataConvert(dataSet):
    radar_map = [[0 for i in range(101)] for i in range(101)]
    newDataSet = []
    for k in range(60):
        for j in range(101):
            for i in range(101):
                radar_map[j][i] = dataSet[k*101*101 + j*101 +i]

        for n in range(51):
            new = 0
            if(n == 0):
                new = radar_map[50][50]

            else:
                for i in range(2*n+1):
                    new += sum(radar_map[50-n + i][50-n:50+n+1])
            newDataSet.append(new)
            #newDataSet.append(new/(n+1))
            newDataSet.append(new/ (n+1)**2)
    return newDataSet


def train():
    dataSet_all = []
    rainfall_all = []
    print("aaaaaaaaaaaaaaaaa")
    print("bbbbbbbbbbbbbbbb")
    for i in range(8):
        testName, rainfall, dataSet = getDataSet1(i, 'train')
        dataSet_all.extend(dataSet)
        rainfall_all.extend(rainfall)

    print("cccccccccccccccc")
    gbdt = ensemble.GradientBoostingRegressor( )
    gbdt.fit(dataSet_all, rainfall_all)  # 训练数据来学习，不需要返回值
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    joblib.dump(gbdt, "rain_train_GBDT_feature_1.m")
    print("ddddddddddddddd")


def predict():
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    gbdt = joblib.load("rain_train_GBDT_feature_1.m")
    for i in range(2):
        print("1")
        testName, rainfall, dataSet = getDataSet1(i, 'test')
        print("2")
        print("3")
        y = gbdt.predict(dataSet)
        print("4")
        save_file = "E:\\李卓聪\\save_file\\rainfall_predict\\" + "feature_1_" + str(i) + ".txt"
        fl = open(save_file, 'w')
        for k in y:
            fl.write(str(k))
            fl.write("\n")
        fl.close()
        print("5")
    print("6")


def crossVali():
    os.chdir("E:\\李卓聪\\save_file\\machine_model")
    gbdt = joblib.load("rain_train_GBDT_feature_1.m")
    dataSet_all = []
    rainfall_all = []
    score_all = 0
    for i in range(2):
        testName, rainfall, dataSet = getDataSet1(i+8, 'train')


        y = gbdt.predict(dataSet)

        score =  y - rainfall
        score = np.sum(np.square(score))
        score_all += score
    score_all = score_all / 2000
    score_all = score_all**0.5
    print(score_all)




if __name__ == "__main__":
    train()
    crossVali()
    #predict()