#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../include/util.h"
#include "../include/Matrix.h"

namespace util {
	int process_img(const char* path, std::vector<Matrix> &x_train, std::vector<std::vector<double>> &y_train, unsigned int num_classes) {
		std::string file(path);

		const int width = 28;
		const int height = 28;
		const int labels = 10;

		for (size_t i = 0; i < labels; i++) {
			std::vector<cv::String> files;
			cv::glob(file + std::to_string(i), files, true);
			for(size_t k = 0; k < ((int)num_classes/labels); k++) {
				cv::Mat img = cv::imread(files[k]);
				if(img.empty()) continue;
				Matrix image(width, height);
				for(size_t h = 0; h < height; h++) {
					for(size_t w = 0; w < width; w++) {
						image.set(h, w, double(img.at<uchar>(h, w))/255.0);
					}
				}
				x_train.push_back(image);
				// x_train.push_back(*image);
				// std::unique_ptr<std::vector<double>> lbl = std::make_unique<std::vector<double>>(labels, 0.0);
				// lbl->at(i) = 1.0;
				std::vector<double> lbl(labels, 0.0);
				lbl[i] = 1.0;
				y_train.push_back(lbl);
			}
		}
		return 0;
	}


	int process_csv(const char* path, std::vector<std::vector<double>> &x_train, std::vector<std::vector<double>> &y_train, bool test) {
		std::string pathName(path);
		std::ifstream file(pathName);
		// create a tokenizer to parse the line
		std::string::const_iterator it;
		std::string line;
		while(std::getline(file, line)) {
			std::stringstream lineStream(line);
			it = line.begin();
			char lblVal = *it;
			int label = std::stoi(std::string(1, lblVal));
			std::vector<double> labelVec(10, 0.0);
			// one hot encode the values by index
			labelVec[label] = 1.0;
			std::vector<double> pixels;
			std::string pixel;
			lineStream.seekg(2, std::ios_base::beg);
			while(std::getline(lineStream, pixel, ',')) {
				pixels.push_back(std::stod(pixel)/255.0);
			}
			x_train.push_back(pixels);
			y_train.push_back(labelVec);
		}
		if(test){ 
			printf("y_train \n");
			for(auto &vec : y_train) {
				printf("[ ");
				for(auto &val: vec) {
					printf("%.1f ", val);
				}
				printf("]\n");
			}
			printf("\n");
		}
		return 0;
	}

	int process_csv(const char* path, std::vector<std::vector<double>> &x_val, std::vector<double> &y_val) {
		std::string pathName(path);
		std::ifstream file(pathName);
		if(!file.is_open()) {
			std::cout << "Error opening file" << std::endl;
			return -1;
		}
		std::string::const_iterator it;
		x_val.clear();
		std::string line;
		while(std::getline(file, line)) {
			std::stringstream stream(line);
			it = line.begin();
			auto val = *it;
			int label = std::stoi(std::string(1, val));
			std::vector<double> pixels;
			std::string pixel;
			// set the stream to the next character
			stream.seekg(2, std::ios_base::beg);
			while(std::getline(stream, pixel, ',')) {
				pixels.push_back(std::stod(pixel)/255.0);
			}
			x_val.push_back(pixels);
			y_val.push_back(label);
		}
		return 0;
	}


	int vecToMat(std::vector<double> &vec, cv::Mat &mat, int w, int h)
	{
		mat = cv::Mat(w, h, CV_8UC1);
		for (size_t i = 0; i < w; i++)
		{
			for (size_t j = 0; j < h; j++)
			{
				mat.at<uchar>(i, j) = vec[i * h + j] * 255;
			}
		}
		if (mat.empty())
			return -1;
		return 0;
	}

} // namespace util