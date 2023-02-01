#pragma once

#include <math.h>
#include <random>
#include <vector>
#include "Activation.h"
#include "Matrix.h"

class Neuron
{
private:
	Matrix wgts;
	Matrix bs;
	double (*activation)(double);
	double (*d_activation)(double);
public:
	Neuron(int numInputs);
	Neuron(Matrix weights, Matrix bias);
	~Neuron();

	Matrix getWeights() const;
	Matrix getBias() const;
	
	double forward(Matrix &inputs, double (*activation)(double));
	double backward(Matrix &inputs, double (*d_activation)(double));
	void optimize(double epsilon);
};
