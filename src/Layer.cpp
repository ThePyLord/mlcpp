#include <random>
#include <cassert>
#include <ctime>
#include <iostream>
#include "../include/Layer.h"
#include "../include/Activation.h"
#include "../include/Matrix.h"
#include "../include/linalg.h"


static size_t timesPrinted = 0;

Layer::Layer(int numInputs, int layerSize) : numInputs(numInputs),
											 layerSize(layerSize),
											 weights(numInputs, layerSize, "weights"),
											 bias(1, layerSize, "bias"),
											 dL_dW(1, 1),
											 dL_db(1, 1),
											 x(1, 1),
											 y(1, 1) {}

Layer::~Layer() {}

Matrix Layer::forward(Matrix &x, double (*actFn)(double)) {
	Matrix prod{np::dot(x, weights)};
	Matrix z = prod + bias;
	y = np::applyFn(z, actFn);
	// storing inputs for backprop
	this->x = x;
	return y;
}

Matrix Layer::backward(Matrix &dl_dy, double (*d_actFn)(double)) {
	// Matrix dy_dz{np::applyFn(dl_dy, d_actFn)};
	Matrix dy_dz = np::applyFn(dl_dy, d_actFn);
	Matrix dL_dz = dl_dy * dy_dz;
	Matrix dz_dw = np::transpose(x);
	Matrix dz_dx = np::transpose(weights);
	Matrix dz_db = np::ones(1, dl_dy.shape[0]);
	Matrix dL_dx = np::dot(dL_dz, dz_dx);

	dL_dW = np::dot(dz_dw, dL_dz);
	dL_db = np::dot(dz_db, dL_dz);
	return dL_dx;
}

Matrix Layer::getWeights()
{
	return weights;
}

void Layer::optimize(double epsilon)
{
	weights = weights - (epsilon * dL_dW);
	bias = bias - (epsilon * dL_db);
}

void Layer::printWeights()
{
	// if (timesPrinted < 1) {
		std::cout << "weights: " << std::endl;
		std::cout << weights << std::endl;
		std::cout << "bias: " << std::endl;
		std::cout << bias << std::endl;
		timesPrinted++;
	// }
}