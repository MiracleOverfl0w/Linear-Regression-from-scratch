#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include<math.h>
#include<stdlib.h>
#define DEFAULT_COFF 0.0f
#define DEFAULT_YINTERCEPT 1.0f
#define DEFAULT_LEARNING_RATE 0.0001f

typedef struct info_lr{
	double coff;
	double Y_intercept;
	double Learning_rate;
	double MSE;
}info_lr;

typedef struct model_lr{
	info_lr info;
	double *X_train,*Y_train;
	int size_dataset;	
}model_lr;


void LrReg(model_lr * model,info_lr init,double *_init_setX,double *_init_setY,int ndata);
void calculateMSE(model_lr *  model);
void grad(model_lr *  model);
void train(model_lr * model,size_t epochs);
double *predict(model_lr * model, double * _dataset,int nsize);
void plot(model_lr * model);


#endif
