#include <iostream>
#include <ctime>
#include <chrono>
#include <thread>
#include <cassert>
#include <algorithm>
#include "../include/linalg.h"
#include "../include/NeuralNet.h"

static size_t timesPrinted = 0;

NeuralNet::NeuralNet(int numInputs, std::vector<int> hiddenLayerSizes, int numOutputs) : 
numInputs(numInputs),
hiddenLayerSizes(hiddenLayerSizes),
numOutputs(numOutputs)
{
	std::vector<int> sizes; // sizes of each layer
	sizes.push_back(numInputs);
	printf("NeuralNet dims: [%d, ", numInputs);
	for (int i = 0; i < (int)hiddenLayerSizes.size(); i++)
	{
		printf("%d, ", hiddenLayerSizes[i]);
		sizes.push_back(hiddenLayerSizes[i]);
	}
	printf("%d]\n", numOutputs);
	sizes.push_back(numOutputs);

	for (size_t i = 0; i < sizes.size() - 1; i++)
	{
		layers.push_back(Layer(sizes[i], sizes[i + 1]));
	}
}

NeuralNet::~NeuralNet() {}

Matrix NeuralNet::forward(Matrix &inputs, double (*actFn)(double))
{
	// Matrix result = inputs;
	Matrix result(inputs);

	for (auto &&layer : layers) {
		result = layer.forward(result, actFn);
	}
	return result;
}

Matrix NeuralNet::backward(Matrix &inputs, double (*actFn)(double))
{
	Matrix res(inputs);

	for (int i = layers.size() - 1; i >= 0; i--)
	{
		res = layers[i].backward(res, actFn);
	}
	
	return res;
}

Tuple NeuralNet::train(
	std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train,
	std::vector<std::vector<double>> &x_val, std::vector<double> &y_val,
	bool validate, int batchSize, int epochs, double alpha)
{
	assert(x_train.size() == y_train.size());
	int batchesPerEpoch = x_train.size() / batchSize;
	std::vector<double> loss, accuracy;
	for (int i = 0; i < epochs; i++)
	{
		double epochLoss = 0;
		auto start = std::chrono::high_resolution_clock::now();
		for (int b = 0; b < batchesPerEpoch; b++)
		{
			int b_idx = b * batchSize; // start of the batch
			int b_end = b_idx + batchSize; // end of the batch
			// Create batches of the data
			std::vector<std::vector<double>> xBatch(&x_train[b_idx], &x_train[b_end]);
			std::vector<std::vector<double>> yBatch(&y_train[b_idx], &y_train[b_end]);

			// shuffle the batches
			// if(validate) {
			// 	auto gen2 = generator;
			// 	std::shuffle(xBatch.begin(), xBatch.end(), generator);
			// 	std::shuffle(yBatch.begin(), yBatch.end(), gen2);
			// }

			Matrix x(xBatch);
			Matrix targets(yBatch, "labels");

			Matrix predictions = forward(x);
			double currLoss = fns::loss_l2(predictions, targets);
			Matrix dL_dy = fns::dLossL2(predictions, targets);
			backward(dL_dy, fns::dSigmoid);
			optimize(alpha);

			// update the loss
			epochLoss += currLoss;
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		epochLoss /= batchesPerEpoch;
		loss.push_back(epochLoss);
		if (validate)
		{
			double acc = eval_accuracy(x_val, y_val);
			accuracy.push_back(acc);
			printf("Epoch: %4d/%d: Loss = %.6f | Validation accuracy: %.2f%%, Time: %ldms\n", i + 1, epochs, loss[i], acc * 100, duration.count());
		}
		else
			accuracy.push_back(NAN);
	}
	std::cout << "Training complete!" << std::endl;
	return std::make_tuple(loss, accuracy);
}

Tuple NeuralNet::train(
	std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train,
	std::vector<std::vector<double>> &x_val, std::vector<std::vector<double>> &y_val,
	bool validate,
	int batchSize, int epochs, double epsilon)
{
	assert(x_train.size() == y_train.size());
	int batchesPerEpoch = x_train.size() / batchSize;
	std::vector<double> loss, accuracy;
	for (int i = 0; i < epochs; i++)
	{
		double epochLoss = 0;
		auto start = std::chrono::high_resolution_clock::now();
		for (int b = 0; b < batchesPerEpoch; b++)
		{
			int bIdxStart = b * batchSize;
			int bIdxEnd = bIdxStart + batchSize;

			std::vector<std::vector<double>> xBatch(&x_train[bIdxStart], &x_train[bIdxEnd]);
			std::vector<std::vector<double>> yBatch(&y_train[bIdxStart], &y_train[bIdxEnd]);

			Matrix x(xBatch);
			Matrix y(yBatch);

			Matrix y_pred = forward(x);
			double currLoss = fns::loss_l2(y_pred, y);
			Matrix dL_dy = fns::dLossL2(y_pred, y);

			backward(dL_dy, fns::dSigmoid);
			optimize(epsilon);

			// update the loss
			epochLoss += currLoss;
		}
		epochLoss /= batchesPerEpoch;
		loss.push_back(epochLoss);
		// timing the batch computation
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		if (validate)
		{
			double acc = eval_accuracy(x_val, y_val);
			accuracy.push_back(acc);
			printf("Epoch: %4d/%d: Loss = %.6f | Validation accuracy: %.2f%% | Time: %ldms\n", i + 1, epochs, epochLoss, acc * 100, duration.count());
		}
		else
			accuracy.push_back(NAN);
	}
	return std::make_tuple(loss, accuracy);
}

double NeuralNet::cross_entropy(std::vector<double> &y, std::vector<double> &y_hat)
{
	double loss = 0;
	for (int i = 0; i < (int)y.size(); i++)
	{
		loss += y[i] * log(y_hat[i]);
	}
	return -loss;
}

double NeuralNet::cross_entropy(Matrix &y, Matrix &y_hat)
{
	assert(y.numRows() == y_hat.numRows() && y.numCols() == y_hat.numCols());
	double loss = 0;
	for (int i = 0; i < y.numRows(); i++)
	{
		for (int j = 0; j < y.numCols(); j++)
		{
			loss += y.get(i, j) * log(y_hat.get(i, j));
		}
	}
	return -loss;
}

void NeuralNet::optimize(double epsilon)
{
	for (auto &&layer : layers)
	{
		layer.optimize(epsilon);
	}
}

int NeuralNet::predict(Matrix &x)
{
	Matrix estimations = forward(x);
	int best_class = np::argmax(estimations);
	return best_class;
}

double NeuralNet::eval_accuracy(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y)
{
	assert(x.size() == y.size());
	double numCorrect{0};

	for (int i = 0; i < (int)y.size(); i++)
	{
		Matrix xMat(x[i]);
		Matrix yMat(y[i]);
		int predicted = predict(xMat);
		int actual = np::argmax(yMat);
		if (predicted == actual)
			numCorrect++;
	}
	return numCorrect / (double)x.size();
}

double NeuralNet::eval_accuracy(std::vector<std::vector<double>> &x, std::vector<double> &y)
{
	assert(x.size() == y.size());
	double numCorrect{0};

	for (size_t i = 0; i < x.size(); i++)
	{
		Matrix xMat(x[i]);
		int predicted = predict(xMat);
		auto y_i = y[i];
		// printf("y[i]: %.3f\n", y[i]);
		if (predicted == static_cast<int>(y[i]))
			numCorrect++;
	}
	return numCorrect / static_cast<double>(x.size());
}