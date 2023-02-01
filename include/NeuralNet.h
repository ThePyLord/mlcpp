#pragma once

#include <random>
#include <vector>
#include <tuple>

#include "Layer.h"
#include "Activation.h"
#include "util.h"

// random number generator for shuffling data
static std::default_random_engine generator(time(0)); 
typedef std::tuple<std::vector<double>, std::vector<double>> Tuple;

class NeuralNet
{
private:
	std::vector<Layer> layers;
	int numInputs;
	std::vector<int> hiddenLayerSizes;
	int numOutputs;
public:
	NeuralNet(int numInputs, std::vector<int> hiddenLayerSizes, int numOutputs);
	~NeuralNet();

	Matrix forward(Matrix &inputs, double (*actFn)(double)=fns::sigmoid);
	Matrix backward(Matrix &inputs, double (*actFn)(double));
	
	void optimize(double alpha);

	/**
	 * @brief Predicts the label of the input
	 * @param x The input
	 * @return The predicted label
	*/
	int predict(Matrix &x);
	
	/**
	 * @brief Evaluates the accuracy of the model
	 * @param x The data
	 * @param y The labels
	 * @return The accuracy of the model
	*/
	double eval_accuracy(std::vector<std::vector<double>> &x, std::vector<double> &y);
	/**
	 * @brief Evaluates the accuracy of the model
	 * @param x The data
	 * @param y The labels(one-hot encoded)
	 * @return The accuracy of the model
	*/
	double eval_accuracy(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y);
	
	/**
	 * @brief Train the neural network
	 * @param x_train The training data
	 * @param y_train The training labels
	 * @param x_val The validation data
	 * @param y_val The validation labels
	 * @param validate Whether to validate the model
	 * @param batchSize Batch size(default is 32)
	 * @param epochs Number of epochs(default is 10)
	 * @param alpha Learning rate(default is 5e-3)
	*/
	Tuple train(
		std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train, 
		std::vector<std::vector<double>> &x_val, std::vector<double> &y_val, 
		bool validate=true,
		int batchSize=32, int epochs=10, double alpha=5e-3);


	Tuple train(
		std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train, 
		std::vector<std::vector<double>> &x_val, std::vector<std::vector<double>> &y_val, 
		bool validate=true,
		int batchSize=32, int epochs=10, double alpha=5e-3);

	double cross_entropy(std::vector<double> &y, std::vector<double> &y_hat);
	double cross_entropy(Matrix &y, Matrix &y_hat);

};
