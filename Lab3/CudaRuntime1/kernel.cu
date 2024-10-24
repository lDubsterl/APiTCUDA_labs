#include <iostream>
#include "D:\Soft\CUDA Toolkit\include\cuda_runtime.h"
#include "D:\Soft\CUDA Toolkit\include\device_launch_parameters.h"
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// Функция для создания матрицы с случайными значениями
std::vector<std::vector<float>> create_random_matrix(int rows, int cols) {
	std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			matrix[i][j] = dis(gen);

	return matrix;
}

// Ядро CUDA для отражения по вертикали
__global__ void flip_gpu(float* matrix, float* result, int rows, int cols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // индекс текущего потока
	int step = cols; // шаг между элементами в строке
	int new_cols = cols * 2; // количество столбцов в результирующей матрице
	int new_rows = rows / 2;

	// Вычисляем координаты строки и столбца
	int row = idx / cols;    // исходная строка
	int col = idx % cols;    // исходный столбец

	if (row < new_rows) {
		// Расчет позиции в результирующей матрице
		int res_idx_1 = row * new_cols + col * 2; // первая позиция в результирующей строке
		int res_idx_2 = row * new_cols + col * 2 + 1; // вторая позиция в результирующей строке

		// Копируем элементы
		result[res_idx_1] = matrix[row * 2 * step + col];       // Элемент из текущей строки
		result[res_idx_2] = matrix[(row * 2 + 1) * step + col]; // Элемент из следующей строки
	}
}

// Функция для вывода части матрицы (N элементов)
void print_partial_matrix(const std::vector<std::vector<float>>& matrix, int N) {
	for (int i = 0; i < matrix.size() && i < N; ++i) {
		for (int j = 0; j < matrix[0].size() && j < N; ++j) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

// Функция для вывода части одномерного массива как двумерной матрицы
void print_partial_flat_matrix(const std::vector<float>& matrix, int rows, int cols, int N) {
	for (int i = 0; i < rows && i < N; ++i) {
		for (int j = 0; j < cols && j < N; ++j) {
			std::cout << matrix[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
}

// Функция для замера времени выполнения на CPU
void measure_cpu(const std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& result) {
	int rows = matrix.size();
	int cols = matrix[0].size();
	int step = cols / 2; // Размер шага между блоками

	// Перестановка блоков
	int new_rows = rows / 2;  // Количество строк в новой матрице
	int new_cols = cols * 2;  // Количество столбцов в новой матрице

	// Инициализация результирующей матрицы с новыми размерами
	result = std::vector<std::vector<float>>(new_rows, std::vector<float>(new_cols, 0));

	auto start = std::chrono::high_resolution_clock::now();

	// Перестановка блоков
	for (int row = 0; row < new_rows; ++row) {
		for (int i = 0; i < cols; ++i) {
			// Копируем два элемента из текущей строки и два элемента из следующей строки
			result[row][2 * i] = matrix[2 * row][i];       // Элемент из текущей строки
			result[row][2 * i + 1] = matrix[2 * row + 1][i];  // Элемент из следующей строки
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;

	std::cout << "CPU Time: " << duration.count() << " seconds" << std::endl;
}

// Функция для замера времени выполнения на GPU
static void measure_gpu(float* d_matrix, float* d_result_matrix, int rows, int cols) {
	int total_size = rows * cols;
	int threads_per_block = 256;
	int blocks = (total_size + threads_per_block - 1) / threads_per_block;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Отражение по вертикали
	flip_gpu << <blocks, threads_per_block >> > (d_matrix, d_result_matrix, rows, cols);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "GPU Time: " << milliseconds / 1000.0 << " seconds" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Полное поэлементное сравнение матриц
bool compare_results(const std::vector<std::vector<float>>& cpu_matrix, const std::vector<float>& gpu_matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (fabsf(cpu_matrix[i][j] - gpu_matrix[i * cols + j]) > 1e-5) {
				return false;
			}
		}
	}
	return true;
}

int main() {
	int rows = 24576, cols = 24576;
	size_t matrix_size = rows * cols * sizeof(float);

	// Создание матрицы на CPU
	auto cpu_matrix = create_random_matrix(rows, cols);

	// Копирование данных в линейный массив для GPU
	std::vector<float> flat_cpu_matrix(rows * cols);
	for (int i = 0; i < rows; ++i)
		std::copy(cpu_matrix[i].begin(), cpu_matrix[i].end(), flat_cpu_matrix.begin() + i * cols);

	// Выделение памяти на GPU
	float* d_matrix;
	float* d_matrix_res;
	cudaMalloc(&d_matrix, matrix_size);
	cudaMalloc(&d_matrix_res, matrix_size);

	// Копирование данных на GPU
	cudaMemcpy(d_matrix, flat_cpu_matrix.data(), matrix_size, cudaMemcpyHostToDevice);

	// Замер времени на CPU
	std::vector<std::vector<float>> cpu_result(rows, std::vector<float>(cols));
	measure_cpu(cpu_matrix, cpu_result);

	// Замер времени на GPU
	measure_gpu(d_matrix, d_matrix_res, rows, cols);

	// Получение результатов с GPU
	std::vector<float> gpu_result(rows / 2 * cols * 2);
	cudaMemcpy(gpu_result.data(), d_matrix_res, matrix_size, cudaMemcpyDeviceToHost);

	// Сравнение результатов
	if (compare_results(cpu_result, gpu_result, rows / 2, cols * 2)) {
		std::cout << "CPU and GPU results match!" << std::endl;
	}
	else {
		std::cout << "Results differ!" << std::endl;
	}

	// Вывод части матрицы с CPU
	std::cout << "Partial Matrix CPU:" << std::endl;
	print_partial_matrix(cpu_result, 10);

	// Вывод части матрицы с GPU
	std::cout << "\n\nPartial Matrix GPU:" << std::endl;
	print_partial_flat_matrix(gpu_result, rows / 2, cols * 2, 10);

	// Освобождение памяти
	cudaFree(d_matrix);
	cudaFree(d_matrix_res);

	return 0;
}
