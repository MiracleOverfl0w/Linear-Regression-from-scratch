#include "LinearRegression.h"

void LrReg(model_lr * model,info_lr init,double *_init_setX,double *_init_setY,int ndata){
	model->info = (info_lr){
		init.coff,
		init.Y_intercept,
		init.Learning_rate
	};
	model->X_train = _init_setX;
	model->Y_train = _init_setY;
	model->size_dataset = ndata;
}
void calculateMSE(model_lr *  model){
	double sigma = 0.0,calculate = 0.0;
	int n = model->size_dataset;
	for(int i = 0;i < n;i++){
		calculate = model->Y_train[i]-(model->info.coff * model->X_train[i] + model->info.Y_intercept);
		sigma += calculate*calculate;
	}

	model->info.MSE = sigma/n;
}
void grad(model_lr * model){
	int n = model->size_dataset;
	double sum_yinter = 0.0,sum_coff = 0.0;
	for(int i = 0;i < n;i++){
		sum_yinter += (model->Y_train[i] - (model->info.coff * model->X_train[i] + model->info.Y_intercept));
		sum_coff += model->X_train[i] * (model->Y_train[i] - (model->info.coff * model->X_train[i] + model->info.Y_intercept));
	}
	double factor = -2.0 / n;
	model->info.coff -= model->info.Learning_rate * factor * sum_coff;
	model->info.Y_intercept -= model->info.Learning_rate * factor * sum_yinter;
	
}
void train(model_lr * model,size_t epochs){
	for(int i=0;i<epochs;i++) grad(model);
	calculateMSE(model);	
}
double * predict(model_lr * model, double * _dataset,int nsize){

	double *_dest = (double*) malloc(sizeof(double)*nsize);

	for(int i = 0;i < nsize;i++) _dest[i] = model->info.coff * _dataset[i] + model->info.Y_intercept;
	return _dest;
}
void plot(model_lr * model);
