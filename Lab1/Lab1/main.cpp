#include <iostream>
#include <chrono>
#include <vector>

using namespace std::chrono;

const int MATRIX_SIZE = sqrt(16 * 1024 * 1024 / sizeof(float));
const int BLOCK_SIZE = sqrt(4 * 1024 * 1024 / sizeof(float));

//const int MATRIX_SIZE = 4;
//const int BLOCK_SIZE = 2;

void clean_memory(float** matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		delete matrix[i];
	}

	delete matrix;
}

float** generate_matrix()
{
	float** matrix = new float* [MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		matrix[i] = new float[MATRIX_SIZE];
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			matrix[i][j] = (float)(rand() % 10); /// 10;
		}
	}

	return matrix;
}

void print_matrix(float** matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			std::cout << matrix[i][j] << " ";
		}

		std::cout << '\n';
	}
}

void matrix_multiplication_1(float** matrix_1, float** matrix_2, float** result)
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			float result_elem = 0.0f;
			for (int k = 0; k < MATRIX_SIZE; k++)
			{
				result_elem += matrix_1[i][k] * matrix_2[k][j];
			}

			result[i][j] = result_elem;
		}
	}
}

//
//float** mul(float** matrix_1, int start_x, int start_y, float** matrix_2, int start2_x, int start2_y, float** result) {
//	float res = 0;
//
//	for (int i = 0; i < BLOCK_SIZE; i++)
//	{
//		for (int j = 0; j < BLOCK_SIZE; j++)
//		{
//			for (int k = 0; k < BLOCK_SIZE; k++)
//			{
//				res += matrix_1[start_x + i][start_x + k] * matrix_2[start2_x + j][start2_y + k];
//			}
//
//			//result[x][y] += res;
//		}
//	}
//}

void blocks_handling(float** matrix_1, float** matrix_2, float** result, int matrix_1_row_offset, int matrix_1_column_offset, int matrix_2_row_offset, int matrix_2_column_offset, int result_row_offset, int result_column_offset)
{
	for (int i = 0; i < BLOCK_SIZE; i++)
	{
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			float result_elem = 0.0f;
			for (int k = 0; k < BLOCK_SIZE; k++)
			{
				result_elem += matrix_1[i + matrix_1_row_offset][k + matrix_1_column_offset] * matrix_2[k + matrix_2_row_offset][j + matrix_2_column_offset];
			}

			result[i + result_row_offset][j + result_column_offset] += result_elem;
		}
	}
}

void matrix_multiplication_2(float** matrix_1, float** matrix_2, float** result)
{	
	//blocks_handling(matrix_1, matrix_2, result, 0, 0, 0, 0, 0, 0); // 0 0 FIX
	//blocks_handling(matrix_1, matrix_2, result, 0, BLOCK_SIZE, 0, BLOCK_SIZE, 0, BLOCK_SIZE); // 0 1
	//blocks_handling(matrix_1, matrix_2, result, BLOCK_SIZE, 0, 0, 0, BLOCK_SIZE, 0); // 1 0
	//blocks_handling(matrix_1, matrix_2, result, BLOCK_SIZE, BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE); // 1 1 FIX
	for (int i = 0; i < MATRIX_SIZE; i += BLOCK_SIZE)
	{
		for (int j = 0; j < MATRIX_SIZE; j += BLOCK_SIZE)
		{
			for (int k = 0; k < MATRIX_SIZE; k += BLOCK_SIZE)
			{
				blocks_handling(matrix_1, matrix_2, result, i, k, k, j, i, j);
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

int main()
{
	srand(time(NULL));
	float** matrix_1 = generate_matrix();
	float** matrix_2 = generate_matrix();
	float** result = new float* [MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		result[i] = new float[MATRIX_SIZE];
	}

	//print_matrix(matrix_1, MATRIX_SIZE);
	std::cout << "-------------------\n";
	//print_matrix(matrix_2, MATRIX_SIZE);
	std::cout << "-------------------\n";
	high_resolution_clock::time_point start = high_resolution_clock::now();
	matrix_multiplication_1(matrix_1, matrix_2, result);
	time_calculation(start);
	//print_matrix(result, MATRIX_SIZE);
	std::cout << "-------------------\n";
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			result[i][j] = 0;
		}
	}

	start = high_resolution_clock::now();
	matrix_multiplication_2(matrix_1, matrix_2, result);
	time_calculation(start);
	//print_matrix(result, MATRIX_SIZE);
	return 0;
}
