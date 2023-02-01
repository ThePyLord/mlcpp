#include <cassert>
#include "../include/linalg.h"

namespace np
{
	auto zeros(int rows, int cols) -> Matrix
	{
		Matrix m(rows, cols);
		return m;
	}

	auto ones(int rows, int cols) -> Matrix
	{
		Matrix m(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m.set(i, j, 1);
			}
		}
		return m;
	}

	Matrix identity(size_t size)
	{
		std::vector<std::vector<double>> identityMat(size, std::vector<double>(size, 0.0));
		for(size_t i = 0; i < size; i++) {
			identityMat[i][i] = 1.0;
		}
		return Matrix(identityMat);
	}

	Matrix random(int rows, int cols)
	{
		Matrix m(rows, cols);
		std::mt19937 generator;
		generator.seed(time(0));
		std::uniform_real_distribution<double> distribution(-1.0, 1.0);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				m.set(i, j, distribution(generator));
			}
		}
		return m;
	}

	Matrix transpose(Matrix &m)
	{
		Matrix res(m.numCols(), m.numRows());
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				res.set(j, i, m.get(i, j));
			}
		}
		return res;
	}

	int argmax(Matrix &m)
	{
		double max = m.get(0, 0);
		int maxIndex = 0;
		std::vector<double> maxIndices;
		int mIdx;
		for(auto &&row : m.values()) {
			mIdx = std::max_element(row.begin(), row.end()) - row.begin();
		}
		int max_index = 0;
		double max_value = m.get(0, 0);

		for (int i = 0; i < m.shape[0]; i++) {
			for (int j = 0; j < m.shape[1]; j++) {
				if (m.get(i, j) > max_value) {
					max_value = m.get(i, j);
					max_index = j;
				}
			}
		}
		return max_index;
		return mIdx;
	}

	double sum(const Matrix &m1)
	{
		double res{0};
		for(const auto& row: m1.values()) {
			res += std::accumulate(row.begin(), row.end(), 0.0);
		}
		return res;
	}

	double multiply(Matrix &m1, Matrix &m2) {
		// assert(m1.numCols() == m2.numRows());
		double sum{0};
		for (int i = 0; i < m1.numRows(); i++) {
			for (int j = 0; j < m2.numCols(); j++) {
				sum += m1.get(i, j) * m2.get(i, j);
			}
		}
		return sum;
	}

	Matrix multiply(Matrix &m1, double &val)
	{
		Matrix res(m1.numRows(), m1.numCols());
		for (int i = 0; i < m1.numRows(); i++) {
			for (int j = 0; j < m1.numCols(); j++) {
				res.set(i, j, m1.get(i, j) * val);
			}
		}
		return res;
	}

	Matrix square(const Matrix &m1)
	{
		std::vector<std::vector<double>> data(m1.shape[0], std::vector<double>(m1.shape[1]));
		for(int i = 0; i < m1.shape[0]; i++) {
			for(int j = 0; j < m1.shape[1]; j++) {
				data[i][j] = m1.get(i, j) * m1.get(i, j);
			}
		}
		return Matrix(data);
	}

	Matrix subtract(Matrix &m1, Matrix &m2)
	{
		Matrix result;
		// assert(m1.numCols() == m2.numCols() && m1.numRows() == m2.numRows());
		std::vector<std::vector<double>> data;
		if(m1.numRows() != m2.numRows() or m1.numCols() != m2.numCols()) {
		int maxRows = std::max(m1.numRows(), m2.numRows());
		int maxCols = std::max(m1.numCols(), m2.numCols());
		for (int i = 0; i < maxRows; i++) {
			std::vector<double> row;
			for (int j = 0; j < maxCols; j++) {
				double val1 = m1.get(i % m1.numRows(), j % m1.numCols());
				double val2 = m2.get(i % m2.numRows(), j % m2.numCols());
				row.push_back(val1 - val2);
			}
			data.push_back(row);
		}
		result = Matrix(data, "result");
		}
		else {
			for (int i = 0; i < m1.numRows(); i++) {
				std::vector<double> row;
				for (int j = 0; j < m1.numCols(); j++) {
					row.push_back(m1.get(i, j) - m2.get(i, j));
				}
				data.push_back(row);
			}
			result = Matrix(data, "result");
		}
		return result;
	}

	Matrix subtract(Matrix &m, double x) {
		Matrix res(m.numRows(), m.numCols());
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				res.set(i, j, m.get(i, j) - x);
			}
		}
		return res;
	}

	Matrix dot(Matrix &m1, Matrix &m2) {
		assert(m1.numCols() == m2.numRows());
		Matrix result(m1.numRows(), m2.numCols());
		std::vector<std::vector<double>> data;
		for (int i = 0; i < m1.numRows(); i++)
		{
			std::vector<double> row;
			for (int j = 0; j < m2.numCols(); j++) {
				double sum{0};
				for (int k = 0; k < m1.numCols(); k++) {
					sum += m1.get(i, k) * m2.get(k, j);
				}
				// row.push_back(sum);
				result.set(i, j, sum);
			}
			data.push_back(row);
		}
	
		assert(result.numRows() == m1.numRows() && result.numCols() == m2.numCols());
		// printf("Actual size: (%d, %d)\n", mat.numRows(), mat.numCols());
		return result;
	}

	Matrix norm(const Matrix &m1)
	{
		std::vector<std::vector<double>> data(m1.shape[0], std::vector<double>(m1.shape[1]));
		for(int i = 0; i < m1.shape[0]; i++) {
			for(int j = 0; j < m1.shape[1]; j++) {
				data[i][j] = m1.get(i, j) / m1.get(i, j);
			}
		}
		return Matrix(data);
	}

	double normalize(const Matrix &m1)
	{
		double norm = 0;
		for(int i = 0; i < m1.shape[0]; i++) {
			for(int j = 0; j < m1.shape[1]; j++) {
				// norm += m1.get(i, j) * m1.get(i, j);
				norm += std::pow(std::abs(m1.get(i, j)), 2);
			}
		}
		norm = std::pow(norm, 1.0 / 2.0);
		return norm;
	}

	std::unique_ptr<Matrix> reshape(Matrix &m, int rows, int cols)
	{
		assert(rows * cols == m.size);

		std::vector<std::vector<double>> newData;
		for (int i = 0; i < rows; i++)
		{
			std::vector<double> row;
			for (int j = 0; j < cols; j++)
			{
				row.push_back(m.get(i, j));
			}
			newData.push_back(row);
		}
		return std::make_unique<Matrix>(newData);
	}

	std::unique_ptr<Matrix> flatten(Matrix &m) {
		Matrix res(1, m.numCols());
		int k = 0;
		for(int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				res.set(0, k++, m.get(i, j));
			}
		}
		return std::make_unique<Matrix>(res);
	}


	std::unique_ptr<Matrix> concatenate(Matrix &m1, Matrix &m2)
	{
		assert(m1.numRows() == m2.numRows());
		Matrix result(m1.numRows(), m1.numCols() + m2.numCols());
		for (int i = 0; i < m1.numRows(); i++) {
			for (int j = 0; j < m1.numCols(); j++) {
			
				result.set(i, j, m1.get(i, j));
			}
			for (int j = 0; j < m2.numCols(); j++) {
				result.set(i, j + m1.numCols(), m2.get(i, j));
			}
		}
		return std::make_unique<Matrix>(result);
	}

	std::unique_ptr<Matrix> concatenate(Matrix &m1, Matrix &m2, int axis)
	{
		assert(axis == 0 or axis == 1);
		std::unique_ptr<Matrix> mat;
		if (axis == 0)
		{
			mat = concatenate(m1, m2);
		}
		else
		{
			Matrix result(m1.numRows() + m2.numRows(), m1.numCols());
			for (int i = 0; i < m1.numRows(); i++)
			{
				for (int j = 0; j < m1.numCols(); j++) {
					result.set(i, j, m1.get(i, j));
				}
			}
			for (int i = 0; i < m2.numRows(); i++) {
				for (int j = 0; j < m2.numCols(); j++) {
					result.set(i + m1.numRows(), j, m2.get(i, j));
				}
			}
			mat.reset(new Matrix(result));
		}
		return mat;
	}

	std::unique_ptr<Matrix> concatenate(Matrix &m1, std::vector<double> &v)
	{
		assert(m1.numCols() == (int)v.size());
		Matrix result(m1.numRows() + 1, m1.numCols());
		for (int i = 0; i < m1.numRows(); i++)
		{
			for (int j = 0; j < m1.numCols(); j++) {
				result.set(i, j, m1.get(i, j));
			}
		}
		for (int i = 0; i < (int)v.size(); i++) {
			result.set(m1.numRows(), i, v[i]);
		}
		return std::make_unique<Matrix>(result);
	}

	std::unique_ptr<Matrix> slice(Matrix &m, int start, int end) {
		assert(start >= 0 and end <= m.size);
		Matrix result(m.numRows(), end - start);
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < end - start; j++) {
				result.set(i, j, m.get(i, j + start));
			}
		}
		return std::make_unique<Matrix>(result);
	}

	Matrix applyFn(Matrix &m, double( *fn)(double))
	{
		Matrix result(m.numRows(), m.numCols());
		for (int i = 0; i < m.numRows(); i++)
		{
			for (int j = 0; j < m.numCols(); j++) {
				result.set(i, j, fn(m.get(i, j)));
			}
		}
		return result;
	}
}
