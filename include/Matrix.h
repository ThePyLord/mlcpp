#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <memory>

// deprecated
struct Shape {
	int rows;
	int cols;
	std::array<int, 2> shape;
	Shape(int rows, int cols) {
		this->rows = rows;
		this->cols = cols;
		this->shape = {rows, cols};
	}
	friend std::ostream& operator<<(std::ostream& os, const Shape& s) {
		os << s.shape[0] << "x" << s.cols;
		os << std::endl;
		return os;
	}
};

class Matrix
{
private:
	int rows;
	int cols;
	std::vector<std::vector<double>> data;
	std::string name;
	double max; // just for debugging purposes
	double min; // for debugging
public:
	int size;
	std::array<int, 2> shape;
	// set default constructor 
	Matrix() = default;
	Matrix(int rows, int cols, std::string name = "unnamed");
	Matrix(const std::vector<double> &data, std::string name = "unnamed");
	Matrix(const std::vector<std::vector<double>> &data, std::string name = "unnamed");
	Matrix(const Matrix &mat, std::string name = "unnamed");
	~Matrix();
	
	double get(int row, int col) const;
	void set(int row, int col, double value);
	void setName(std::string newName);
	std::string getName() const;
	int numRows();
	int numCols();
	std::vector<double> getRows(int row);
	std::vector<double> getCols(int col);
	std::vector<std::vector<double>> values() const;
	void shuffle();


	// operators
	friend Matrix operator+(const Matrix &m1, const Matrix &m2);
	friend Matrix operator+(const Matrix &m1);
	friend Matrix operator-(const Matrix &m1, const Matrix &m2);
	friend Matrix operator*(Matrix &m1, Matrix &m2);
	friend Matrix operator*(Matrix &m1, double& num);
	friend Matrix operator*(double &num, Matrix& m1);
	friend bool operator==(const Matrix &m1, const Matrix &m2);
	friend std::ostream& operator<<(std::ostream &os, const Matrix &m);
};

