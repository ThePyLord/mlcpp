#include <iostream>
#include <vector>
#include <random>
#include "../include/Neuron.h"
#include "../include/linalg.h"
// #include <math.h>
// #include <ctime>

#include "../include/Matrix.h"

Neuron::Neuron(int numInputs) : wgts(1, numInputs), bs(1, 1) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);
	for(int i = 0; i < wgts.size; i++) {
		wgts.set(i, 0, distribution(generator));
	}
	bs.set(0, 0, distribution(generator));
	// printf("Creating a neuron with %d inputs and %s activation\n", numInputs, activation.c_str());
}


Neuron::Neuron(Matrix weights, Matrix bias) : 
	wgts(weights), bs(bias)
{
	printf("Creating a neuron with %d inputs and\n", weights.size);
}

Neuron::~Neuron() {
	printf("Destroying a neuron\n");
}

Matrix Neuron::getWeights() const {
	return wgts;
}

Matrix Neuron::getBias() const {
	return bs;
}

double Neuron::forward(Matrix &inputs, double (*activation_fn)(double)) {
	Matrix z = wgts * inputs;
	z = z + bs;
	for(int i = 0; i < z.size; i++) {
		for(int j = 0; j < z.numCols(); j++) {
			z.set(i, j, (*activation_fn)(z.get(i, j)));
			// z.set(i, j, activation_fn(z.get(i, j)));
		}
	}
	// double result = activation_fn(z);
	double result = z.get(0, 0);
	std::cout << "z: " << z << std::endl;
	return result;
}

double Neuron::backward(Matrix &inputs, double (*d_activation_fn)(double)) {
	double result{0.0};
	Matrix z = wgts * inputs;
	z = z + bs;
	Matrix k = np::applyFn(z, d_activation_fn);
	std::cout << "z: " << z << std::endl;
	return result;
}

void Neuron::optimize(double epsilon) {
	wgts = wgts * epsilon;
	bs = bs * epsilon;
}
