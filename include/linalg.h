#pragma once

#include <random>
#include <algorithm>
#include <numeric>
#include "Matrix.h"

namespace np
{
	Matrix zeros(int rows, int cols);
	Matrix ones(int rows, int cols);
	Matrix identity(size_t size);
	Matrix random(int rows, int cols);
	Matrix dot(Matrix &m1, Matrix &m2);
	Matrix norm(const Matrix &m1);
	double normalize(const Matrix &m1);

	double sum(const Matrix &m1);
	double multiply(Matrix &m1, Matrix &m2);
	Matrix multiply(Matrix &m1, double &val);
	Matrix square(const Matrix &m1);
	Matrix subtract(Matrix &m1, Matrix &m2);
	Matrix subtract(Matrix &m1, double x);
	Matrix transpose(Matrix &m);
	int argmax(Matrix &m);
	
	std::unique_ptr<Matrix> reshape(Matrix &m, int rows, int cols);
	std::unique_ptr<Matrix> flatten(Matrix &m);
	std::unique_ptr<Matrix> concatenate(Matrix &m1, Matrix &m2, int axis);
	std::unique_ptr<Matrix> concatenate(Matrix &m1, Matrix &m2);
	std::unique_ptr<Matrix> concatenate(Matrix &m1, std::vector<double> &v2);
	std::unique_ptr<Matrix> slice(Matrix &m, int start, int end);
	Matrix applyFn(Matrix &x, double (*fn)(double)); // apply function to each element of matrix
} // namespace np
