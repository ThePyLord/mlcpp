#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include "include/util.h"
#include "include/NeuralNet.h"
#include "include/matplotlibcpp.h"

using namespace cv;
namespace plt = matplotlibcpp;

void showImage() {
	Mat img = imread("data/road.png");
	if(img.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
		std::cin.get(); //wait for any key press
		exit(1);
	}

	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", img);
	waitKey(0);
	destroyAllWindows();
}

int main(int argc, char const *argv[])
{
	std::vector<std::vector<double>> xTrain, yTrain, xTest;
	// std::vector<std::vector<double>> yTest;
	std::vector<double> yTest;
	std::cout << "Processing training data...\n";
	util::process_csv("data/mnist_train.csv", xTrain, yTrain);
	std::cout << "Processing test data...\n";
	util::process_csv("data/mnist_test.csv", xTest, yTest);
	// std::cout << "Creating network...\n";

	NeuralNet nn(xTrain[0].size(), {64, 32}, 10);
	double accuracy = nn.eval_accuracy(xTest, yTest);
	printf("Accuracy before training: %.2f%%\n", accuracy * 100);
	std::vector<double> loss, acc;
	// showImage();

	Tuple tup = nn.train(xTrain, yTrain, xTest, yTest, false);
	std::tie(loss, acc) = tup;
	std::cout << "Loss: " << Matrix(loss) << std::endl;
	std::cout << "Accuracy: " << Matrix(acc) << std::endl;
	// std::cout << "Gonna plot now...\n";
	// plt::suptitle("Loss and accuracy");
	// plt::subplot(2, 1, 1);
	// plt::plot(loss);
	// plt::subplot(2, 1, 2);
	// plt::plot(acc);
	// plt::show();


	return 0;
}
