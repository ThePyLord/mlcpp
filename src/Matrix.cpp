#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cassert>
// #include <memory>
#include <algorithm>
#include "../include/Matrix.h"


Matrix::Matrix (int rows, int cols, std::string name): 
rows(rows), 
cols(cols), 
name(name),
size(rows * cols)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	// std::default_random_engine gen(rd());
	std::normal_distribution<double> distribution(0.0, 1.0);
	for (int i = 0; i < rows; i++)
	{
		std::vector<double> row;
		for (int j = 0; j < cols; j++)
		{
			row.push_back(distribution(gen));
			if(row[j] > max) max = row[j];
			if(row[j] < min) min = row[j];
		}
		data.push_back(row);
	}
	std::array<int, 2> s{{rows, cols}};
	shape = s;
	// size = rows * cols;
}

Matrix::Matrix (const std::vector<std::vector<double>> &data, std::string name): 
 shape{{(int)data.size(), (int)data[0].size()}},
 data(data),
 name(name),
 size(data.size() * data[0].size())
{
	if(data.size() == 1) {
		rows = 1;
		cols = data[0].size();
	} else {
		rows = data.size();
		cols = data[0].size();
	}
	size = rows * cols;
	double max = -__DBL_MAX__;
	double min = __DBL_MAX__;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if(data[i][j] > max) max = data[i][j];
			if(data[i][j] < min) min = data[i][j];
		}
	}
	this->max = max;
	this->min = min;
}

Matrix::Matrix(const std::vector<double> &data, std::string name): 
	data{{data}}, 
	name(name),
	shape({{1, (int)data.size()}}), 
	size(data.size() * 1)
{
	rows = 1;
	cols = data.size();
	max = -__DBL_MAX__;
	min = __DBL_MAX__;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if(this->data[i][j] > max) max = this->data[i][j];
			if(this->data[i][j] < min) min = this->data[i][j];
		}
	}
}

Matrix::Matrix(const Matrix &mat, std::string name)
{
	data = mat.data;
	shape = mat.shape;
	rows = mat.rows;
	cols = mat.cols;
	(*this).name = mat.name;
	size = mat.size;
}

Matrix::~Matrix() {}

int Matrix::numRows() {
	return rows;
}

int Matrix::numCols() {
	return cols;
}

double Matrix::get(int row, int col) const
{
	if(row >= rows or col >= cols) return (double)0;
	return data[row][col];
}

void Matrix::set(int row, int col, double value)
{
	if(row >= rows or col >= cols) return;
	data[row][col] = value;
}

void Matrix::setName(std::string newName)
{
	this->name = newName;
}

std::string Matrix::getName() const
{
	return this->name;
}
std::vector<double> Matrix::getRows(int row)
{
	assert(row < rows and row >= 0);
	return data[row];
}

std::vector<double> Matrix::getCols(int col) {
	assert(col < cols);
	std::vector<double> result;
	for (int i = 0; i < rows; i++)
	{
		result.push_back(data[i][col]);
	}
	return result;
}

std::vector<std::vector<double>> Matrix::values() const
{
	return data;
}

void Matrix::shuffle()
{
	for(auto& row: data) {
		std::mt19937 generator;
		generator.seed(std::random_device{}());
		std::shuffle(row.begin(), row.end(), generator);
	}
}

// OPERATORS
Matrix operator+(const Matrix &m1, const Matrix &m2) {
	// assert(m1.rows == m2.rows and m1.cols == m2.cols);
	std::vector<std::vector<double>> data;
	// use the broadcast method if the matrices are not the same size
	if(m1.rows != m2.rows or m1.cols != m2.cols) {
		int maxRows = std::max(m1.rows, m2.rows);
		int maxCols = std::max(m1.cols, m2.cols);
		data.resize(maxRows);
		for (int i = 0; i < maxRows; i++) {
			data[i].resize(maxCols);
			for (int j = 0; j < maxCols; j++) {
				double val1 = m1.data[i % m1.rows][j % m1.cols];
				double val2 = m2.data[i % m2.rows][j % m2.cols];
				data[i][j] = val1 + val2;
			}
		}
	} else {
		data.resize(m1.rows);
		for (int i = 0; i < m1.rows; i++) {
			data[i].resize(m1.cols);
			for (int j = 0; j < m1.cols; j++) {
				data[i][j] = m1.data[i][j] + m2.data[i][j];
			}
		}
	}
	Matrix result = Matrix(data, "result");
	return result;
}

Matrix operator+(const Matrix &m1)
{
	// assert(rows == m1.rows and cols == m1.cols);
	Matrix result(m1.rows, m1.cols);
	for (int i = 0; i < m1.rows; i++)
	{
		for (int j = 0; j < m1.cols; j++)
		{
			result.data[i][j] = m1.data[i][j];
		}
	}
	return result;
}

Matrix operator-(const Matrix &m1, const Matrix &m2)
{
	// printf("Subtracting %s and %s\n", m1.name.c_str(), m2.name.c_str());
	assert(m1.rows == m2.rows and m1.cols == m2.cols);
	Matrix result(m1.rows, m1.cols);
	std::vector<std::vector<double>> data;
	for (int i = 0; i < m1.rows; i++) {
		std::vector<double> row;
		for (int j = 0; j < m1.cols; j++) {
			row.push_back(m1.data[i][j] - m2.data[i][j]);
		}
		data.push_back(row);
	}
	result = Matrix(data, "result");

	return result;
}
Matrix operator*(Matrix &m1, Matrix &m2)
{
	// Do element wise multiplication
	// printf("Element wise multiplying %s and %s\n", m1.name.c_str(), m2.name.c_str());	
	// assert(m1.rows == m2.rows and m1.cols == m2.cols);
	Matrix res(m1.rows, m1.cols);
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			res.data[i][j] = m1.data[i][j] * m2.data[i][j];
		}
	}
	assert(res.rows == m1.rows and res.cols == m2.cols);
	return res;
}

Matrix operator*(Matrix &m1, double& scalar) {
	Matrix result(m1.numRows(), m1.numCols());
	for (int i = 0; i < m1.numRows(); i++)
	{
		for (int j = 0; j < m1.numCols(); j++)
		{
			result.set(i,j, m1.get(i, j) * scalar); 
		}
	}
	return result;
}

Matrix operator*(double &scalar, Matrix& m1) {
	Matrix result(m1.numRows(), m1.numCols());
	for (int i = 0; i < m1.numRows(); i++)
	{
		for (int j = 0; j < m1.numCols(); j++)
		{
			result.set(i,j, m1.get(i, j) * scalar); 
		}
	}
	return result;
}

bool operator==(const Matrix &m1, const Matrix &m2)
{
	assert(m1.shape == m2.shape);
	bool equal = true;
	for(int i = 0; i < m1.shape[0]; i++) {
		for (int j = 0; j < m1.shape[1]; j++)
		{
			if(m1.data[i][j] != m2.data[i][j])
				equal = false;
		}
		
	}
	return equal;
}

std::ostream &operator<<(std::ostream &os, const Matrix &m)
{
	os << "[";
	if(m.cols > 1) os << "\n";
	for (int i = 0; i < m.rows; i++)
	{
		os << " [ ";
		for (int j = 0; j < m.cols; j++)
		{
			if (m.data[i][j] < 0 || m.data[i][j] > 99) {
				std::setprecision(2);
				// Print the double in scientific notation
				os << std::scientific << m.data[i][j] << " ";
			} else {
				// Print the double with default formatting
				os << m.data[i][j] << " ";
			}

		}
		os << "] " << std::endl;
	}
	os << "]" << std::endl;
	return os;
}
