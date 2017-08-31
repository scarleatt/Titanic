from numpy import *

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        del(lineArr[3]); del(lineArr[3]); del(lineArr[7]); del(lineArr[8])
        lineArr[3] = {
            'male': '0',
            'female': '1'
        }.get(lineArr[3], -1)
        lineArr[-1] = {
            'C': '1',
            'Q': '2',
            'S': '3'
        }.get(lineArr[-1], -1)
        labelMat.append(int(lineArr[1]))
        del(lineArr[0]); del(lineArr[1])
        lineArr[3] = int(lineArr[3])+int(lineArr[4])
        del(lineArr[4])
        dataMat.append(lineArr)
    m,n = shape(dataMat)
    sum = 0
    for i in range(m):
        if dataMat[i][2] == '':
            dataMat[i][2] = 0
        sum += float(dataMat[i][2])
    sum /= m
    for i in range(m):
        if dataMat[i][2] == 0:
            dataMat[i][2] = sum
        dataMat[i] = map(float, dataMat[i])
    return dataMat, labelMat

def loadDataSet2(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def dataNormal(dataMat):
    m, n = shape(dataMat)
    for j in range(n):
        if (j == 0): continue
        minValue = min(dataMat[:, j])
        maxValue = max(dataMat[:, j])
        dataMat[:, j] = (dataMat[:, j]-minValue) / (maxValue - minValue)
        # print dataMat[:,j]
    # for i in range(m):
        # print dataMat[i][0][0]
    return dataMat

def sigmod(inx):
    return 1.0/(1+exp(-inx))

def classifyVector(inX, weights):
    prob = sigmod(sum(inX*weights))
    if prob > 0.5: return 1
    else: return 0

def grandAscent(dataMat, classLabels):
    m, n = shape(dataMat)
    alpha = 0.001
    weights = ones(n)
    for k in range(m):
        h = sigmod(sum(dataMat[k]*weights))
        error = classLabels[k] - h
        weights = weights + alpha * error * dataMat[k]
    return weights

def grandAscent1(dataMat, classLabels, numIter=150):
    m, n = shape(dataMat)
    weights = ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4/(i+j+1.0) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(sum(dataMat[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

def selectRandJ(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def smo(dataMatIn, labelMatIn, C, tol, maxIter):
    dataMat = mat(dataMatIn); labelMat = mat(labelMatIn).transpose()
    b = 0; passes = 0
    m, n = shape(dataMat)
    alpha = mat(zeros((m,1)))
    while (passes < maxIter):
        alphaChanged = 0
        for i in range(m):
            fxi = float(multiply(alpha,labelMat).T*(dataMat*dataMat[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if ((labelMat[i]*Ei < -tol) & (alpha[i] < C)) | ((labelMat[i]*Ei > tol) & (alpha[i] > 0)):
                j = selectRandJ(i, m)
                fxj = float(multiply(alpha,labelMat).T*(dataMat*dataMat[j,:].T)) + b
                Ej = fxj - labelMat[j]
                alphaIold = alpha[i].copy(); alphaJold = alpha[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alpha[j]-alpha[i])
                    H = min(C, C+alpha[j]-alpha[i])
                else:
                    L = max(0, alpha[i]+alpha[j]-C)
                    H = min(C, alpha[i]+alpha[j])
                if L == H: continue
                eta = 2.0*dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                if eta >= 0: continue
                alpha[j] = alpha[j] - labelMat[j]*(Ei-Ej)/eta

                if alpha[j] > H: alpha[j] = H
                elif alpha[j] < L: alpha[j] = L

                if abs(alpha[i]-alpha[j]) < 0.00001: continue
                alpha[i] = alpha[i] + labelMat[i]*labelMat[j]*(alphaJold-alpha[j])
                b1 = b-Ei-labelMat[i]*(alpha[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alpha[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T
                b2 = b-Ej-labelMat[i]*(alpha[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alpha[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T

                if (alpha[i] > 0) & (alpha[i] < C): b = b1
                elif (alpha[j] > 0) & (alpha[j] < C): b = b2
                else: b = (b1+b2)/2
                alphaChanged += 1
        if alphaChanged == 0:
            passes += 1
        else:
            passes = 0
    return b, alpha

def colicTest():
    trainMat, trainLabels = loadDataSet('./data/train.csv')
    weights = grandAscent(array(trainMat), trainLabels)
    dataMat = []
    fr = open('./data/test.csv')
    sequences = []
    s = "PassengerId,Survived\n"
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        del(lineArr[2]); del(lineArr[2]); del(lineArr[6]); del(lineArr[-2])
        lineArr[2] = {
            'male': '0',
            'female': '1'
        }.get(lineArr[2], -1)
        lineArr[-1] = {
            'C': '1',
            'Q': '2',
            'S': '3'
        }.get(lineArr[-1], -1)
        sequences.append(int(lineArr[0]))
        dataMat.append(lineArr)
    m,n = shape(dataMat)
    sumAge = 0; sumTmp = 0
    for i in range(m):
        if dataMat[i][3] == '':
            dataMat[i][3] = 0
        sumAge += float(dataMat[i][3])
        if dataMat[i][6] == '':
            dataMat[i][6] = 0
        sumTmp += float(dataMat[i][6])
    sumAge /= m
    for i in range(m):
        if dataMat[i][3] == 0:
            dataMat[i][3] = sumAge
        if dataMat[i][6] == 0:
            dataMat[i][6] = sumTmp
        dataMat[i] = map(float, dataMat[i])
        del(dataMat[i][0])
        dataMat[i][4] += dataMat[i][5]; del(dataMat[i][5])
        s += str(sequences[i])+","+str(classifyVector(array(dataMat[i]), weights))+"\n"
    file=open('result.txt','w')
    file.write(s)
    file.close()

colicTest()
# trainMat, trainLabels = loadDataSet('./data/train.csv')

# print mat(trainMat)
# b, alphas = smoSimple(trainMat, trainLabels, 0.6, 0.001, 40)
# print b
# print alphas[alphas>0]