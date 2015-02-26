#include <sys/time.h>   
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <string>

#pragma offload_attribute(push,target(mic))
//#include <cilk/cilk.h>

#define DataType float
#define INF HUGE_VAL
#define PHIDEV 1
#define VectorLength 16
#define MY_THREADS 240
#define ScheduleState static
#define ALIGNLENGTH 128

struct Kernel_params{
	DataType gamma;
	DataType coef0;
	int degree;
	DataType b;
	std::string kernel_type;
};

DataType parameterA = -0.125;
DataType parameterB = 1.0;
DataType parameterC = 3.0;
DataType cost = 1.0;
DataType tolerance = 1e-3;
DataType epsilon = 1e-5; 
DataType* data;
DataType* labels;
DataType* alpha;
DataType* tempF1;
DataType* tempF2;
DataType * data_square;
DataType* devF;
DataType* devKernelDiag;
DataType* highKernel;
DataType* lowKernel;
DataType cEpsilon, alphaHighDiff, alphaLowDiff, bLow, bHigh, alphaHighOld, alphaLowOld, eta;
Kernel_params kp;
int nPoints, nDimension, iLow, iHigh, iteration;

enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};
enum KernelType {LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID};

  static DataType selfKernel(DataType* pointerA) {
    return 1.0;
  }
  static DataType kernel(DataType* pointerA, DataType* pointerB) {
    //DataType accumulant = __sec_reduce_add((pointerA[0:nDimension]-pointerB[0:nDimension])*(pointerA[0:nDimension]-pointerB[0:nDimension]));
    int i;
    /*
	DataType accumulant=0;
	for(i=0;i<nDimension;i++)
    	accumulant+=((pointerA[i]-pointerB[i])*(pointerA[i]-pointerB[i]));
    return exp(parameterA * accumulant);
    */
    DataType accumulant=0,element;//#pragma vector always aligned //#pragma novector
    for(i=0;i<nDimension;i++){
	    element = (pointerA[i]-pointerB[i]);
	    accumulant+=(element*element);
    }    	
    return exp(parameterA * accumulant);
    
  }  

DataType kernel_dot(DataType* pointerA, DataType* pointerB){
	return __sec_reduce_add(pointerA[0:nDimension]*pointerB[0:nDimension]);
}
  
#pragma offload_attribute(pop)
//template<class Kernel>
void initializeArrays() { 	
	for (int index = 0;index < nPoints;index++) {
		devKernelDiag[index] = selfKernel(data + index*nDimension);
		devF[index] = -labels[index];
	}
}

void launchInitialization() {  
    initializeArrays();   
}

//template<class Kernel>
void takeFirstStep() {                                
	eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
	DataType* pointerA = data + iHigh*nDimension;
	DataType* pointerB = data + iLow*nDimension;
	DataType phiAB = kernel(pointerA, pointerB);	
	eta = eta - 2*phiAB;
	alphaLowOld = alpha[iLow];
	alphaHighOld = alpha[iHigh]; 
	//And we know eta > 0
	DataType alphaLowNew = 2/eta; //Just boil down the algebra
	if (alphaLowNew > cost) {
		alphaLowNew = cost;
	}
	//alphaHighNew == alphaLowNew for the first step
	alpha[iLow] = alphaLowNew;
	alpha[iHigh] = alphaLowNew;	
}
void launchTakeFirstStep(int kType) {
     takeFirstStep();
  }

void performTraining(){	
	int kType = GAUSSIAN;
	if (kp.kernel_type.compare(0,3,"rbf") == 0) {
		parameterA = -kp.gamma;
		kType = GAUSSIAN;
		printf("Gaussian kernel: gamma = %f\n", -parameterA);
	} 
	printf("--Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon); 
	cEpsilon = cost - epsilon; 
	
	devKernelDiag = (DataType*)malloc(sizeof(DataType) * nPoints);
	devF = (DataType*)malloc(sizeof(DataType) * nPoints);
	tempF1 = (DataType*)malloc(sizeof(DataType) * nPoints);
	tempF2 = (DataType*)malloc(sizeof(DataType) * nPoints);
	data_square = (DataType*)malloc(sizeof(DataType) * nPoints);
	highKernel = (DataType*)malloc(sizeof(DataType) * nPoints);
	lowKernel = (DataType*)malloc(sizeof(DataType) * nPoints);
	
	launchInitialization(); 
	printf("Initialization complete\n");

  	bLow = 1;
  	bHigh = -1;
  	iLow = -1;
  	iHigh = -1;
  	for (int i = 0; i < nPoints; i++) {
    	if (labels[i] < 0) {
      		if (iLow == -1) {
        		iLow = i;
        		if (iHigh > -1) {
          		i = nPoints; //Terminate
        		}
      		}
    	} else {
      		if (iHigh == -1) {
        		iHigh = i;
        		if (iLow > -1) {
          			i = nPoints; //Terminate
        		}
      		}
    	}
  } 
  launchTakeFirstStep(kType);
  alphaLowDiff = alpha[iLow] - alphaLowOld;
  alphaHighDiff = -labels[iHigh] * labels[iLow] * alphaLowDiff;
  printf("Starting iterations\n");
  iteration = 1;
  int data_length = nPoints*nDimension;
  struct timeval start,finish;
  gettimeofday(&start, 0);	
  #pragma offload target(mic:PHIDEV)\
  in(parameterA,cost,tolerance,epsilon,alphaHighDiff, alphaLowDiff)\
  in(data:length(data_length) alloc_if(1))\
  in(labels:length(nPoints) alloc_if(1))\
  inout(alpha:length(nPoints) alloc_if(1))\
  in(devF:length(nPoints) alloc_if(1))\
  in(devKernelDiag:length(nPoints) alloc_if(1))\
  nocopy(tempF1:length(nPoints) alloc_if(1))\
  nocopy(data_square:length(nPoints) alloc_if(1))\
  nocopy(tempF2:length(nPoints) alloc_if(1))\
  nocopy(highKernel:length(nPoints) alloc_if(1))\
  nocopy(lowKernel:length(nPoints) alloc_if(1))
  {
	int i;
	DataType* pointer_row;
	#pragma omp parallel for schedule(ScheduleState) //num_threads(MY_THREADS)	//#pragma simd //vectorlength(VectorLength)
	for(i=0;i<nPoints;i++){
		pointer_row = data + i*nDimension;
		data_square[i] = __sec_reduce_add(pointer_row[0:nDimension]*pointer_row[0:nDimension]);
	}		
	while(bLow > bHigh + 2*tolerance){	  	  
	  #pragma omp parallel for schedule(ScheduleState) //num_threads(MY_THREADS)	//#pragma simd //vectorlength(VectorLength)  
	  #pragma vector always aligned
		  for(i=0;i<nPoints;i++){		  
		  //highKernel[i] = kernel(data + i*nDimension, data + (iHigh)*nDimension);
		  highKernel[i] = exp(parameterA*(data_square[i]+data_square[iHigh] - 2*kernel_dot(data + i*nDimension, data + (iHigh)*nDimension)));		  
		  //lowKernel[i] = kernel(data + i*nDimension, data + (iLow)*nDimension);	
		  lowKernel[i] = exp(parameterA*(data_square[i]+data_square[iLow] - 2*kernel_dot(data + i*nDimension, data + (iLow)*nDimension)));	  
		  devF[i] = devF[i] + alphaHighDiff * labels[iHigh] * highKernel[i] + alphaLowDiff * labels[iLow] * lowKernel[i];

		  if(((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon)))
			  	tempF1[i]=devF[i];
			  else
			  	tempF1[i]=INF;
			  	
		  if(((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon)))
			  	tempF2[i]=devF[i];
			  else
			  	tempF2[i]=-INF;
	  	}
	  
	  //tempF[0:nPoints] = (((labels[0:nPoints] > 0) && (alpha[0:nPoints] < cEpsilon)) || ((labels[0:nPoints] < 0) && (alpha[0:nPoints] > epsilon))) ? devF[0:nPoints] : INF;
	  
	  iHigh = __sec_reduce_min_ind(tempF1[0:nPoints]);
	  
	  //tempF[0:nPoints] = (((labels[0:nPoints] > 0) && (alpha[0:nPoints] > epsilon)) || ((labels[0:nPoints] < 0) && (alpha[0:nPoints] < cEpsilon))) ? devF[0:nPoints] : -INF;
      
	  iLow = __sec_reduce_max_ind(tempF2[0:nPoints]);
	  bHigh = devF[iHigh];
	  bLow = devF[iLow];
	  //eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2 * kernel(data + iHigh*nDimension, data + iLow*nDimension);  
	  eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2 * exp(parameterA*(data_square[iHigh]+data_square[iLow]-2*kernel_dot(data + iHigh*nDimension, data + iLow*nDimension))); 
	  alphaHighOld = alpha[iHigh];
	  alphaLowOld = alpha[iLow];
	  DataType alphaDiff = alphaLowOld - alphaHighOld;
	  DataType lowLabel = labels[iLow];
	  DataType sign = labels[iHigh] * lowLabel;
	  DataType alphaLowUpperBound;
	  DataType alphaLowLowerBound;
	  if (sign < 0) {
          if (alphaDiff < 0) {
              alphaLowLowerBound = 0;
              alphaLowUpperBound = cost + alphaDiff;
          } else {
			  alphaLowLowerBound = alphaDiff;
              alphaLowUpperBound = cost;
		  }
      } else {
		DataType alphaSum = alphaLowOld + alphaHighOld;
		if (alphaSum < cost) {
			alphaLowUpperBound = alphaSum;
			alphaLowLowerBound = 0;
		} else {
			alphaLowLowerBound = alphaSum - cost;
			alphaLowUpperBound = cost;
        }
	  }

      DataType alphaLowNew;
      if (eta > 0) {
		alphaLowNew = alphaLowOld + lowLabel*(bHigh - bLow)/eta;
		if (alphaLowNew < alphaLowLowerBound) alphaLowNew = alphaLowLowerBound;
        else if (alphaLowNew > alphaLowUpperBound) alphaLowNew = alphaLowUpperBound;
	  } else {
		DataType slope = lowLabel * (bHigh - bLow);
		DataType delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
		if (delta > 0) {
			if (slope > 0)	alphaLowNew = alphaLowUpperBound;
			else	alphaLowNew = alphaLowLowerBound;
		} else	alphaLowNew = alphaLowOld;
	  }
    alphaLowDiff = alphaLowNew - alphaLowOld;
    alphaHighDiff = -sign*(alphaLowDiff);
    alpha[iLow] = alphaLowNew;
    alpha[iHigh] = (alphaHighOld + alphaHighDiff);
	  	iteration++;
	  	//printf("iLow: %d, bLow: %lf; iHigh: %d, bHigh: %lf\n",iLow,bLow,iHigh,bHigh);
	  	//if(iteration==30)
	  	//	break;
	  }
  }
  gettimeofday(&finish, 0);
  DataType trainingTime = (DataType)(finish.tv_sec - start.tv_sec) + ((DataType)(finish.tv_usec - start.tv_usec)) * 1e-6;	
  printf("Training time : %f seconds\n", trainingTime);
  printf("Converged\n");
  printf("--- %d iterations ---\n", iteration);
  printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
  kp.b = (bLow + bHigh) / 2;
  printf("b: %f\n", kp.b);
  FILE * npk_time = fopen("npk_time1","a+");
  fprintf(npk_time,"%f\n", trainingTime);
}
int readSvm(const char* filename, int* p_npoints, int* p_dimension) {
	FILE* inputFilePointer = fopen(filename, "r");
	if (inputFilePointer == 0) {
		printf("File not found\n");
		return 0;
	}
	int npoints = 0;
	int dimension = 0;
	char c;
	char firstLine = 1;
	do {
		c = fgetc(inputFilePointer);
		switch(c) {
		case '\n':
			npoints++;
			firstLine = 0;
			break;
		case ':':
			if (firstLine > 0) {
				dimension++;
			}
		default:
			;
		}			
	} while (c != EOF);
	rewind(inputFilePointer);
	*(p_npoints) = npoints;
	*(p_dimension) = dimension;	
	
	/*
	data = (DataType*)_mm_malloc(sizeof(DataType)*npoints*dimension, ALIGNLENGTH);
	labels = (DataType*)_mm_malloc(sizeof(DataType)*npoints, ALIGNLENGTH);
	*/
	data = (DataType*)malloc(sizeof(DataType)*npoints*dimension);
	labels = (DataType*)malloc(sizeof(DataType)*npoints);
	char* stringBuffer = (char*)malloc(65536);	
	for(int i = 0; i < npoints; i++) {
		char* bufferPointer = stringBuffer;
		char validCharacter = 1;
		int currentDim = 0;
		int parsingLabel = 1;
		do {
			c = fgetc(inputFilePointer);
			if (validCharacter > 0) {
				if ((c == ' ') || (c == '\n')) {
					*(bufferPointer) = 0;
					DataType value;
					sscanf(stringBuffer, "%f", &value);
					if (parsingLabel > 0) {
						labels[i] = value;
						parsingLabel = 0;
					} else {
						//data[currentDim*npoints + i] = value;
						data[i*dimension + currentDim] = value;
						currentDim++;
					}
					validCharacter = 0;
					bufferPointer = stringBuffer;
				} else {
					*(bufferPointer) = c;
					bufferPointer++;
				}
			}
			if (c == ':') {
				validCharacter = 1;
			}
		} while (c != '\n');
	}				
	free(stringBuffer);
	fclose(inputFilePointer);
	return 1;
}

void printModel(const char* outputFile) { 
	printf("Output File: %s\n", outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) {
		printf("Can't write %s\n", outputFile);
		exit(1);
	}
	int nSV = 0;
	int pSV = 0;
	for(int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			if (labels[i] > 0) {
				pSV++;
			} else {
				nSV++;
			}
		}
	}
  bool printGamma = false;
  bool printCoef0 = false;
  bool printDegree = false;
  const char* kernelType = kp.kernel_type.c_str();
  if (strncmp(kernelType, "polynomial", 10) == 0) {
    printGamma = true;
    printCoef0 = true;
    printDegree = true;
  } else if (strncmp(kernelType, "rbf", 3) == 0) {
    printGamma = true;
  } else if (strncmp(kernelType, "sigmoid", 7) == 0) {
    printGamma = true;
    printCoef0 = true;
  }
	
	fprintf(outputFilePointer, "svm_type c_svc\n");
	fprintf(outputFilePointer, "kernel_type %s\n", kp.kernel_type.c_str());
  if (printDegree) {
    fprintf(outputFilePointer, "degree %i\n", kp.degree);
  }
  if (printGamma) {
    fprintf(outputFilePointer, "gamma %f\n", kp.gamma);
  }
  if (printCoef0) {
    fprintf(outputFilePointer, "coef0 %f\n", kp.coef0);
  }
	fprintf(outputFilePointer, "nr_class 2\n");
	fprintf(outputFilePointer, "total_sv %d\n", nSV + pSV);
	fprintf(outputFilePointer, "rho %.10f\n", kp.b);
	fprintf(outputFilePointer, "label 1 -1\n");
	fprintf(outputFilePointer, "nr_sv %d %d\n", pSV, nSV);
	fprintf(outputFilePointer, "SV\n");
	for (int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			for (int j = 0; j < nDimension; j++) {
				fprintf(outputFilePointer, "%d:%.10f ", j+1, data[i*nDimension + j]);
			}
			fprintf(outputFilePointer, "\n");
		}
	}
	fclose(outputFilePointer);
}

void printHelp() {
  printf("Usage: svmTrain [options] trainingData.svm\n");
  printf("Options:\n");
  printf("\t-o outputFilename\t Location of output file\n");
  printf("Kernel types:\n");
  printf("\t--gaussian\tGaussian or RBF kernel (default): Phi(x, y; gamma) = exp{-gamma*||x-y||^2}\n");
  printf("\t--linear\tLinear kernel: Phi(x, y) = x . y\n");
  printf("\t--polynomial\tPolynomial kernel: Phi(x, y; a, r, d) = (ax . y + r)^d\n");
  printf("\t--sigmoid\tSigmoid kernel: Phi(x, y; a, r) = tanh(ax . y + r)\n");
  printf("Parameters:\n");
  printf("\t-c, --cost\tSVM training cost C (default = 1)\n");
  printf("\t-g\tGamma for Gaussian kernel (default = 1/nDimension)\n");
  printf("\t-a\tParameter a for Polynomial and Sigmoid kernels (default = 1/l)\n");
  printf("\t-r\tParameter r for Polynomial and Sigmoid kernels (default = 1)\n");
  printf("\t-d\tParameter d for Polynomial kernel (default = 3)\n");
  printf("Convergence parameters:\n");
  printf("\t--tolerance, -t\tTermination criterion tolerance (default = 0.001)\n");
  printf("\t--epsilon, -e\tSupport vector threshold (default = 1e-5)\n");
  printf("Internal options:\n");
  printf("\t--heuristic, -h\tWorking selection heuristic:\n");
  printf("\t\t0: First order\n");
  printf("\t\t1: Second order\n");
  printf("\t\t2: Random (either first or second order)\n");
  printf("\t\t3: Adaptive (default)\n");
}

static int kType = GAUSSIAN;

int main( const int argc, const char** argv)  {   	  	
	int currentOption;
  	bool parameterASet = false;
  	bool parameterBSet = false;
  	bool parameterCSet = false;  
  	SelectionHeuristic heuristicMethod = ADAPTIVE; 
  	char* outputFilename = NULL; 
  	
  while (1) {
    static struct option longOptions[] = {
      {"gaussian", no_argument, &kType, GAUSSIAN},
      {"polynomial", no_argument, &kType, POLYNOMIAL},
      {"sigmoid", no_argument, &kType, SIGMOID},
      {"linear", no_argument, &kType, LINEAR},
      {"cost", required_argument, 0, 'c'},
      {"heuristic", required_argument, 0, 'h'},
      {"tolerance", required_argument, 0, 't'},
      {"epsilon", required_argument, 0, 'e'},
      {"output", required_argument, 0, 'o'},
      {"version", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'f'}
    };
    int optionIndex = 0;
    currentOption = getopt_long(argc, (char *const*)argv, "c:h:t:e:o:a:r:d:g:v:f", longOptions, &optionIndex);
    if (currentOption == -1) {
      break;
    }
    int method = 3;
    switch (currentOption) {
    case 0:
      break;
    case 'v':
      printf("MICSVM version: 1.0\n");
      return(0);
    case 'f':
      printHelp();
      return(0);
    case 'c':
      sscanf(optarg, "%f", &cost);
      break;
    case 'h':
      sscanf(optarg, "%i", &method);
      switch (method) {
      case 0:
        heuristicMethod = FIRSTORDER;
        break;
      case 1:
        heuristicMethod = SECONDORDER;
        break;
      case 2:
        heuristicMethod = RANDOM;
        break;
      case 3:
        heuristicMethod = ADAPTIVE;
        break;
      }
      break;
    case 't':
      sscanf(optarg, "%f", &tolerance);
      break;
    case 'e':
      sscanf(optarg, "%f", &epsilon);
      break;
    case 'o':
      outputFilename = (char*)malloc(strlen(optarg));
      strcpy(outputFilename, optarg);
      break;
    case 'a':
      sscanf(optarg, "%f", &parameterA);
      parameterASet = true;
      break;
    case 'r':
      sscanf(optarg, "%f", &parameterB);
      parameterBSet = true;
      break;
    case 'd':
      sscanf(optarg, "%f", &parameterC);
      parameterCSet = true;
      break;
    case 'g':
      sscanf(optarg, "%f", &parameterA);
      parameterA = -parameterA;
      parameterASet = true;
      break;
    case '?':
      break;
    default:
      abort();
      break;
    }
  }

  if (optind != argc - 1) {
    printHelp();
    return(0);
	}

	const char* trainingFilename = argv[optind];  
  
  if (outputFilename == NULL) {
    int inputNameLength = strlen(trainingFilename);
    outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
    strncpy(outputFilename, trainingFilename, inputNameLength + 4);
    char* period = strrchr(outputFilename, '.');
    if (period == NULL) {
      period = outputFilename + inputNameLength;
    }
    strncpy(period, ".mdl\0", 5);
  }  
	
	readSvm(trainingFilename, &nPoints, &nDimension);
	printf("Input data found: %d points, %d dimension\n", nPoints, nDimension); 
	/*
	alpha = (DataType*)_mm_malloc(nPoints*sizeof(DataType), ALIGNLENGTH);
	alpha[0:nPoints] = 0;
	*/
	alpha = (DataType*)calloc(nPoints, sizeof(DataType));
  
  if (kType == LINEAR) {
    printf("Linear kernel\n");
    //kp.kernel_type = "linear";
  	} else if (kType == POLYNOMIAL) {
    if (!(parameterCSet)) {
      parameterC = 3.0f;
    }
    if (!(parameterASet)) {
      //parameterA = 1.0/nPoints;
	  parameterA = 1.0/nDimension;
    }
    if (!(parameterBSet)) {
      parameterB = 0.0f;
    }
    //printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
      printf("Invalid parameters\n");
      exit(1);
    }
    kp.kernel_type = "polynomial";
    kp.gamma = parameterA;
    kp.coef0 = parameterB;
    kp.degree = (int)parameterC;
  	} else if (kType == GAUSSIAN) {
    if (!(parameterASet)) {
      //parameterA = 1.0/nPoints;
	  parameterA = 1.0/nDimension;
    } else {
      parameterA = -parameterA;
    }
    //printf("Gaussian kernel: gamma = %f\n", parameterA);
    if (parameterA < 0) {
      printf("Invalid parameters\n");
      exit(1);
    }
    kp.kernel_type = "rbf";
    kp.gamma = parameterA;
  } else if (kType == SIGMOID) {
    if (!(parameterASet)) {
      //parameterA = 1.0/nPoints;
	  parameterA = 1.0/nDimension;
    }
    if (!(parameterBSet)) {
      parameterB = 0.0f;
    }
    //printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
    if ((parameterA <= 0) || (parameterB < 0)) {
      printf("Invalid Parameters\n");
      exit(1);
    }
    kp.kernel_type = "sigmoid";
    kp.gamma = parameterA;
    kp.coef0 = parameterB;
  }
	struct timeval start,finish;
	gettimeofday(&start, 0);
	performTraining();
	gettimeofday(&finish, 0);
	DataType trainingTime = (DataType)(finish.tv_sec - start.tv_sec) + ((DataType)(finish.tv_usec - start.tv_usec)) * 1e-6;	
	printf("all time : %f seconds\n", trainingTime);
	printModel(outputFilename);
	return 0;
}
