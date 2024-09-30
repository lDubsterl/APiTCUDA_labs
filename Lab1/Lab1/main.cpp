#include <iostream>
#include <chrono>
#include <vector>
#include <intrin.h>
#include <malloc.h>
#include "omp.h"

using namespace std::chrono;

const int MATRIX_SIZE = sqrt(64 * 1024 * 1024 / sizeof(float));
const int BLOCK_SIZE = sqrt(4 * 1024 * 1024 / sizeof(float));

//const int MATRIX_SIZE = 16;
//const int BLOCK_SIZE = 16;

void clean_memory(float** matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		//delete matrix[i];
		_aligned_free(matrix[i]);
	}

	//delete[] matrix;
	_aligned_free(matrix);
}

float** generate_matrix()
{
	//float** matrix = new float* [MATRIX_SIZE];
	float** matrix = (float**)_aligned_malloc(MATRIX_SIZE * sizeof(float*), 32);
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		//matrix[i] = new float[MATRIX_SIZE];
		matrix[i] = (float*)_aligned_malloc(MATRIX_SIZE * sizeof(float), 32);
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			matrix[i][j] = (int)(rand() % 10);
		}
	}

	return matrix;
}

void print_matrix(float** matrix)
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			std::cout << matrix[i][j] << " ";
		}

		std::cout << '\n';
	}
	std::cout << '\n';
}

void matrix_multiplication_1(float** matrix_1, float** matrix_2, float** result)
{
	//for (int i = 0; i < MATRIX_SIZE; i++)
	//{
	//	for (int j = 0; j < MATRIX_SIZE; j++)
	//	{
	//		float result_elem = 0.0f;
	//		for (int k = 0; k < MATRIX_SIZE; k++)
	//		{
	//			result_elem += matrix_1[i][k] * matrix_2[k][j];
	//		}

	//		result[i][j] = result_elem;
	//	}
	//}

	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			for (int k = 0; k < MATRIX_SIZE; k++)
			{
				result[j][k] += matrix_1[j][i] * matrix_2[i][k];
			}
		}
	}
}

void blocks_parallel_handling(float** matrix_1, float** matrix_2, float** result,
	int matrix_1_row_offset, int matrix_1_column_offset, int matrix_2_row_offset, int matrix_2_column_offset,
	int result_row_offset, int result_column_offset)
{
#pragma omp parallel for
	for (int i = 0; i < BLOCK_SIZE; i++)
	{
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			for (int k = 0; k < BLOCK_SIZE; k += 8)
			{
				__m256 a = _mm256_set1_ps(matrix_1[i + matrix_1_row_offset][j + matrix_1_column_offset]);
				__m256 b = _mm256_load_ps(&matrix_2[j + matrix_2_row_offset][k + matrix_2_column_offset]);

				__m256 partialResult = _mm256_load_ps(&result[i + result_row_offset][k + result_column_offset]);
				partialResult = _mm256_fmadd_ps(a, b, partialResult);
				_mm256_store_ps(&result[i + result_row_offset][k + result_column_offset], partialResult);
			}
		}
	}
}

void blocks_handling(float** matrix_1, float** matrix_2, float** result,
	int matrix_1_row_offset, int matrix_1_column_offset, int matrix_2_row_offset, int matrix_2_column_offset,
	int result_row_offset, int result_column_offset, bool is_AVX_used)
{
	__m256 a, b;
	for (int i = 0; i < BLOCK_SIZE; i++)
	{
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			for (int k = 0; k < BLOCK_SIZE; k++)
			{
				if (is_AVX_used)
				{
					a = _mm256_set1_ps(matrix_1[i + matrix_1_row_offset][j + matrix_1_column_offset]);
					b = _mm256_load_ps(&matrix_2[j + matrix_2_row_offset][k + matrix_2_column_offset]);

					__m256 partialResult = _mm256_load_ps(&result[i + result_row_offset][k + result_column_offset]);
					partialResult = _mm256_fmadd_ps(a, b, partialResult);
					_mm256_store_ps(&result[i + result_row_offset][k + result_column_offset], partialResult);

					k += 7;
				}
				else
					result[i + result_row_offset][k + result_column_offset] += matrix_1[i + matrix_1_row_offset][j + matrix_1_column_offset] * matrix_2[j + matrix_2_row_offset][k + matrix_2_column_offset];
			}
		}
	}
}

void matrix_multiplication_2(float** matrix_1, float** matrix_2, float** result, bool is_AVX_used = false, bool is_parallel = false)
{
	for (int i = 0; i < MATRIX_SIZE; i += BLOCK_SIZE)
	{
		for (int j = 0; j < MATRIX_SIZE; j += BLOCK_SIZE)
		{
			for (int k = 0; k < MATRIX_SIZE; k += BLOCK_SIZE)
			{
				if (is_parallel)
					blocks_parallel_handling(matrix_1, matrix_2, result, i, k, k, j, i, j);
				else
					blocks_handling(matrix_1, matrix_2, result, i, k, k, j, i, j, is_AVX_used);
			}
		}
	}
}

void time_calculation(high_resolution_clock::time_point start)
{
	high_resolution_clock::time_point finish = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(finish - start);
	std::cout << "Took " << time_span.count() << " seconds." << std::endl;
}

bool matrix_compare(float** matrix_1, float** matrix_2)
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			if (abs(matrix_1[i][j] - matrix_2[i][j]) > 1e-3)
			{
				std::cout << matrix_1[i][j] << " " << matrix_2[i][j] << i << " " << j << std::endl;
				return false;
			}
		}
	}

	return true;
}

int main()
{
	//srand(time(NULL));
	float** matrix_1 = generate_matrix();
	float** matrix_2 = generate_matrix();

	float** result_1 = (float**)_aligned_malloc(MATRIX_SIZE * sizeof(float*), 32);
	float** result_2 = (float**)_aligned_malloc(MATRIX_SIZE * sizeof(float*), 32);
	float** result_3 = (float**)_aligned_malloc(MATRIX_SIZE * sizeof(float*), 32);

	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		result_1[i] = (float*)_aligned_malloc(MATRIX_SIZE * sizeof(float), 32);
		result_2[i] = (float*)_aligned_malloc(MATRIX_SIZE * sizeof(float), 32);
		result_3[i] = (float*)_aligned_malloc(MATRIX_SIZE * sizeof(float), 32);
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			result_1[i][j] = result_2[i][j] = result_3[i][j] = 0;
		}
	}
	high_resolution_clock::time_point start = high_resolution_clock::now();
	matrix_multiplication_2(matrix_1, matrix_2, result_1);
	time_calculation(start);

	start = high_resolution_clock::now();
	matrix_multiplication_2(matrix_1, matrix_2, result_2, true);
	time_calculation(start);

	start = high_resolution_clock::now();
	matrix_multiplication_2(matrix_1, matrix_2, result_3, true, true);
	time_calculation(start);

	/*print_matrix(result_1);
	print_matrix(result_2);
	print_matrix(result_3);*/

	if (matrix_compare(result_1, result_2) && matrix_compare(result_2, result_3))
	{
		std::cout << "Results are same." << std::endl;
	}
	else
	{
		std::cout << "Results aren't same." << std::endl;
	}

	return 0;
}
