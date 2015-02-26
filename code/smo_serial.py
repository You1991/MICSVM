#!/usr/bin/python
import sys
import numpy as np
import math

def rmi(string):
    p = string.find(':')
    l = len(string)
    return float(string[p+1:l])

def parseVector(line):
    return [rmi(x) for x in line]

def kernel(a, b, n, gamma):
    sum = 0
    for i in range(0, n):
        sum = sum + (a[i]-b[i])**2
    #sum = np.sum((a - b) ** 2)
    return math.exp(-1.0*gamma*sum)

def selfkernel():
    return 1.0
    
if __name__ == "__main__":

    if len(sys.argv) != 5:
        print >> sys.stderr, "Usage: ./smo.py <input_file> <output_file> <cost> <gamma>"
        exit(-1)

    print >> sys.stderr, """SMO implementation by Yang You, only for dense dataset!!!"""

    infile = open(sys.argv[1], "r")
    cost = float(sys.argv[3])
    gamma = float(sys.argv[4])
    print "Cost = " + str(cost) + " Gamma = " + str(gamma)
    data = []
    label = []
    m = 0
    reads = infile.readline()
    n = len(reads.split())-1
    line = reads[0:len(reads)-1]
    while True:
        #print '***********'
        #print line
        if line == '':
            break
        m = m + 1
        sample = line.split(' ')
        label.append(int(sample[0]))
        data.append(parseVector(sample[1:n+1]))
        reads = infile.readline()
        line = reads[0:len(reads)-1]
    tolerance = 1e-3
    epsilon = 1e-5
    cEpsilon = cost - epsilon
    print 'number of samples is: ' + str(m) +  ' number of features is: ' + str(n)
    print 'tolerance is: ' + str(tolerance) + ' epsilon is: ' + str(epsilon)
    #print label
    #print data
    devKernelDiag = []
    devF = []
    tempF1 = []
    tempF2 = []
    #highKernel = []
    #lowKernel = []
    alpha = []

    for i in range(0, m):
        devKernelDiag.append(selfkernel())
        devF.append(-1.0*label[i])
        alpha.append(0)
        #highKernel.append(0)
        #lowKernel.append(0)
        tempF1.append(0)
        tempF2.append(0)
    
    bLow = 1;
    bHigh = -1;
    iLow = -1;
    iHigh = -1;
    for i in range(0, m):
        if (label[i] < 0):
            if (iLow == -1):
                iLow = i
                if (iHigh > -1):
                    i = m
        else:
            if (iHigh == -1):
                iHigh = i
                if (iLow > -1):
                    i = m
    eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
    pHigh = data[iHigh]
    pLow = data[iLow]
    phiAB = kernel(pHigh, pLow, n, gamma);    
    eta = eta - 2*phiAB;
    alphaLowOld = alpha[iLow]
    alphaHighOld = alpha[iHigh]
    if eta != 0:
        alphaLowNew = 2.0/eta
    else:
        alphaLowNew = 0.0

    if (alphaLowNew > cost):
        alphaLowNew = cost
    alpha[iLow] = alphaLowNew
    alpha[iHigh] = alphaLowNew
    alphaLowDiff = alpha[iLow] - alphaLowOld
    alphaHighDiff = -1.0 * label[iHigh] * label[iLow] * alphaLowDiff
    iteration = 1

    while(bLow > bHigh + 2*tolerance):    
        for i in range(0, m):         
            #highKernel[i] = kernel(data[i], data[iHigh], n, gamma);         
            #lowKernel[i] = kernel(data[i], data[iLow], n, gamma);       
            #devF[i] = devF[i] + alphaHighDiff * label[iHigh] * highKernel[i] + alphaLowDiff * label[iLow] * lowKernel[i];
            devF[i] = devF[i] + alphaHighDiff * label[iHigh] * kernel(data[i], data[iHigh], n, gamma) + alphaLowDiff * label[iLow] * kernel(data[i], data[iLow], n, gamma);

            if(((label[i] > 0) and (alpha[i] < cEpsilon)) or ((label[i] < 0) and (alpha[i] > epsilon))):
                tempF1[i] = devF[i];
            else:
                tempF1[i] = float('Inf');

            if(((label[i] > 0) and (alpha[i] > epsilon)) or ((label[i] < 0) and (alpha[i] < cEpsilon))):
                tempF2[i] = devF[i];
            else:
                tempF2[i] = -float('Inf');
    
        #iHigh = __sec_reduce_min_ind(tempF1[0:nPoints])
        iHigh = tempF1.index(min(tempF1))
        #iLow = __sec_reduce_max_ind(tempF2[0:nPoints]);
        iLow = tempF2.index(max(tempF2))
      
        bHigh = devF[iHigh]
        bLow = devF[iLow]
        eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2 * kernel(data[iHigh], data[iLow], n, gamma)
        alphaHighOld = alpha[iHigh]
        alphaLowOld = alpha[iLow]
        alphaDiff = alphaLowOld - alphaHighOld
        lowLabel = label[iLow]
        sign = label[iHigh] * lowLabel

        if (sign < 0):
            if (alphaDiff < 0):
                alphaLowLowerBound = 0;
                alphaLowUpperBound = cost + alphaDiff;
            else:
                alphaLowLowerBound = alphaDiff;
                alphaLowUpperBound = cost;
        else:
            alphaSum = alphaLowOld + alphaHighOld;
            if (alphaSum < cost):
                alphaLowUpperBound = alphaSum;
                alphaLowLowerBound = 0;
            else:
                alphaLowLowerBound = alphaSum - cost;
                alphaLowUpperBound = cost;

        if (eta > 0):
            alphaLowNew = alphaLowOld + lowLabel*(bHigh - bLow)/eta;
            if (alphaLowNew < alphaLowLowerBound):
                alphaLowNew = alphaLowLowerBound;
            elif (alphaLowNew > alphaLowUpperBound): 
                alphaLowNew = alphaLowUpperBound;
        else:
            slope = lowLabel * (bHigh - bLow);
            delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
            if (delta > 0):
                if (slope > 0):  
                    alphaLowNew = alphaLowUpperBound;
                else:
                    alphaLowNew = alphaLowLowerBound;
            else:
                alphaLowNew = alphaLowOld;
        alphaLowDiff = alphaLowNew - alphaLowOld;
        alphaHighDiff = -sign*(alphaLowDiff);
        alpha[iLow] = alphaLowNew;
        alpha[iHigh] = (alphaHighOld + alphaHighDiff);
        iteration = iteration + 1;
    b = (bLow + bHigh) / 2;
    print "total iterations: " + str(iteration) +  " *** bLow: " + str(bLow) + " *** bHigh: " + str(bHigh) + " *** b: " + str(b)
    
    outfile = open(sys.argv[2], "w")
    print "Output File: " + sys.argv[2]
    nSV = 0
    pSV = 0
    for i in range(0, m):
        if (alpha[i] > epsilon):
            if (label[i] > 0):
                pSV = pSV + 1;
            else:
                nSV = nSV + 1;
    printGamma = False;
    printCoef0 = False;
    printDegree = False;
    degree = None
    kernelType = "rbf";
    if (kernelType == "polynomial"):
        printGamma = True;
        printCoef0 = True;
        printDegree = True;
    elif (kernelType == "rbf"):
        printGamma = True;
    elif (kernelType == "sigmoid"):
        printGamma = True;
        printCoef0 = True;
    
    outfile.write("svm_type c_svc\n");
    outfile.write("kernel_type " + str(kernelType) + "\n");
    if (printDegree):
        outfile.write("degree " + str(degree) + "\n");
    if (printGamma): 
        outfile.write("gamma " + str(gamma) + "\n");
    if (printCoef0):
        outfile.write("coef0 " + str(coef0) + "\n");

    outfile.write("nr_class 2\n");
    outfile.write("total_sv " + str(nSV + pSV) + "\n");
    outfile.write("rho " + str(b) + "\n");
    outfile.write("label 1 -1\n");
    outfile.write("nr_sv " + str(pSV) + " " + str(nSV) + "\n");
    outfile.write("SV\n");
    for i in range(0, m):
        if (alpha[i] > epsilon):
            outfile.write(str(label[i]*alpha[i])+" ");
            for j in range(0, n): 
                outfile.write(str(j+1) + ":"+str(data[i][j]) + " ");
            outfile.write("\n");
    outfile.close();
