#pragma once

#include <vector>
#include <random>
#include "Matrix.h"
#include "Activation.h"
#include "Neuron.h"


class Layer
{
private:
	int numInputs;
	int layerSize;
	Matrix weights;
	Matrix bias;
	Matrix dL_dW;
	Matrix dL_db;
	Matrix x;
	Matrix y;
public:
	Layer(int numInputs, int layerSize);
	~Layer();

	Matrix forward(Matrix &x, double (*actFn)(double));
	Matrix backward(Matrix &x, double (*actFn)(double)=fns::relu);
	Matrix getWeights();
	void optimize(double epsilon);
	void printWeights();
};
