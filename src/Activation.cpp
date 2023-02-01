#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include "../include/Activation.h"
#include "../include/linalg.h"

namespace fns {
	double relu(double x) {
		return x <= 0. ? 0. : x;
	}

	double step(double x) {
		return x <= 0 ? 0 : 1;
	}

	double tanh(double x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	double sigmoid(double x) {
		return 1. / (1. + exp(-x));
	}

	double leakyRelu(double x) {
		return x <= 0 ? 0.01 * x : x;
	}

	double softmax(std::vector<double> &x) {
		double sum = 0;
		for (int i = 0; i < (int)x.size(); i++)
		{
			sum += exp(x[i]);
		}

		return exp(x[0]) / sum;
	}

	double softmax(double x)
	{
		return exp(x) / (exp(x) + exp(-x));
	}

	auto randomizeWeights(std::vector<double> &weights) -> void
	{
		std::mt19937 generator;
		generator.seed(time(0));
		std::normal_distribution<double> distribution(-1.0, 1.0);
		for (int i = 0; i < (int)weights.size(); i++)
		{
			weights[i] = distribution(generator);
		}
		// std::srand(std::time(nullptr));
		// for (int i = 0; i < weights.size(); i++)
		// {
		// 	weights[i] = (double)std::rand() / RAND_MAX;
		// }
	}

	double dRelu(double x)
	{
		return (double)(x > 0.0);
	}

	double dSigmoid(double x)
	{
		return x * (1. - x);
	}

	double loss_l2(Matrix &yHat, Matrix &y)
	{
		// double loss{0};
		// for (int i = 0; i < yHat.numRows(); i++) {
		// 	for (int j = 0; j < yHat.numCols(); j++) {
		// 		// loss += pow(yHat.get(i, j) - y.get(i, j), 2);
		// 		loss += (yHat.get(i, j) - y.get(i, j)) * (yHat.get(i, j) - y.get(i, j));
		// 	}
		// }
		// return loss;
		return np::sum(np::square(yHat - y)) / (double)yHat.shape[0];
	}

	double binCrossEntropy(Matrix &yHat, Matrix &y)
	{
		double res{0};

		return 0.0;
	}

	double catCrossEntropy(Matrix &pred, Matrix &target)
	{
		double loss{0};
		for (int i = 0; i < pred.numRows(); i++) {
			for (int j = 0; j < pred.numCols(); j++) {
				if(target.get(i, j) == 1.)
					loss -= target.get(i, j) * std::log(pred.get(i, j));
					// loss += -log(pred.get(i, j));
				// loss += target.get(i, j) * log(pred.get(i, j));
			}
		}
		return loss / pred.numRows();
	}

	Matrix dLossL2(Matrix &yHat, Matrix &y)
	{
		assert(yHat.numRows() == y.numRows() && yHat.numCols() == y.numCols());
		// Matrix dLoss(yHat.numRows(), yHat.numCols());
		std::vector<std::vector<double>> dLossVec(yHat.numRows(), std::vector<double>(yHat.numCols()));
		for (int i = 0; i < yHat.numRows(); i++) {
			for (int j = 0; j < yHat.numCols(); j++) {
				dLossVec[i][j] = 2 * (yHat.get(i, j) - y.get(i, j));
				// dLoss.set(i, j, 2 * (yHat.get(i, j) - y.get(i, j)));
			}
		}
		Matrix dLoss(dLossVec);
		return dLoss;
	}
}
