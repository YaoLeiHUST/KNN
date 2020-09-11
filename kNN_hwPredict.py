from os import listdir
import numpy as np
from kNN import kNN
# from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vec(filename):
    '''
    函数将图像转为向量
    :param filename:文件名
    :return returnVec:返回(1,1024)向量
    '''
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        #读取每一行
        line = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = line[j]
    return returnVec


def hwClassTrain():
    '''
    手写数字识别训练函数
    '''
    #训练集路径
    trainDir = r'D:\VScodePython\机器学习算法\k近邻\手写数字识别\trainingDigits'
    #读取训练集的所有文件名
    trainFileList = listdir(trainDir)
    #训练集训练样本个数
    numOfTrain = len(trainFileList)
    #初始化训练样本标签
    hwLabels = []
    #初始化训练集数据
    trainMat = np.zeros((numOfTrain, 1024))
    for i in range(numOfTrain):
        fileName = trainFileList[i]
        #读取数据
        trainMat[i, :] = img2vec((trainDir + '\\' + fileName))
        #取文件名的第一位为标签 例：0_0.txt
        hwLabels.append(int(fileName.split('.')[0].split('_')[0]))
    #全局变量
    global model
    #使用自己编写的kNN
    model = kNN(trainMat, hwLabels)
    #利用sklearn 构建kNN分类器
    # neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    #拟合模型
    # neigh.fit(trainMat, hwLabels)


def hwClassTest():
    '''
    手写数字识别测试函数
    '''
    #测试集路径
    testDir = r'D:\VScodePython\机器学习算法\k近邻\手写数字识别\testDigits'
    #测试集文件名
    testFileList = listdir(testDir)
    #测试集数目
    numOfTest = len(testFileList)
    errorCount = 0.0
    for i in range(numOfTest):
        fileName = testFileList[i]
        #测试文件实际标签
        testRealLabel = int(fileName.split('.')[0].split('_')[0])
        testVec = img2vec((testDir + '\\' + fileName))
        global model
        #分类
        classfyLabel = model.classfy(testVec, 3)
        print(classfyLabel)
        #利用sklearn分类
        # classfyLabel=neigh.predict(testVec)
        if classfyLabel != testRealLabel:
            errorCount += 1.0
    print('总错误数：%d' % errorCount)
    print('错误率：%f%%' % (errorCount / float(numOfTest) * 100))


def hwClassPredict():
    '''
    手写数字识别预测函数
    '''
    fileDir = input('请输入要预测的文件路径：')
    predictVec = img2vec(fileDir)
    global model
    classfyLabel = model.classfy(predictVec, 3)
    print('预测类型为：%d' % classfyLabel)


hwClassTrain()
hwClassPredict()