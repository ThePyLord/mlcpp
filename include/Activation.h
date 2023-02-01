#pragma once

#include <memory>
#include <cassert>
#include "Matrix.h"

namespace fns
{
	double relu(double x);
	double step(double x);
	double tanh(double x);
	double sigmoid(double x);
	double leakyRelu(double x);
	double softmax(std::vector<double> &x);
	double softmax(double x);

	// auto randomizeWeights(std::vector<double> &weights) -> void;

	double dRelu(double x);
	// double dStep(double x);
	// double dTanh(double x);
	double dSigmoid(double x);
	// double dLeakyRelu(double x);

	/**
	 * @brief l2 regularization
	 * @param yHat predicted value
	 * @param y target value
	 * @return a double indicating the loss
	*/
	double loss_l2(Matrix &yHat, Matrix &y);
	double binCrossEntropy(Matrix &yHat, Matrix &y);
	double catCrossEntropy(Matrix &pred, Matrix &target);
	Matrix dLossL2(Matrix &yHat, Matrix &y);

}
