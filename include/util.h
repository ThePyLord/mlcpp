#pragma once
#include "Matrix.h"
#include <opencv2/opencv.hpp>

namespace util {
	int process_img(const char* path, std::vector<Matrix> &x_train, std::vector<std::unique_ptr<std::vector<double>>> &y_train, unsigned int num_classes);
	int process_img(const char* path, std::vector<Matrix> &x_train, std::vector<std::vector<double>> &y_train, unsigned int num_classes);
	int process_img(const char* path, std::vector<Matrix> &x_train, std::vector<std::vector<double>> &y_train, unsigned int num_classes);
	
	/**
	 * @brief Process a csv file and return the data in a vector of vectors
	 * @param path The path to the csv file
	 * @param x_train The vector of vectors to store the data
	 * @param y_train The vector of vectors to store the labels
	 * @return 0 on success
	*/
	int process_csv(const char* path, std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train, bool test=false);
	int process_csv(const char* path, std::vector<std::vector<double>> &x_train, std::vector<double> &y_train);

	int vecToMat(std::vector<double> &vec, cv::Mat &mat, int w = 28, int h = 28);
}