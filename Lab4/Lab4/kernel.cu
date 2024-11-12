#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

const int N = 1024 * 1024 * 512;

// Функция для создания матрицы с случайными значениями
std::vector<int> create_random_array(int size) {
	std::vector<int> arr(size);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(-1 * N, N);

	for (int i = 0; i < size; ++i)
		arr[i] = dis(gen);

	return arr;
}

// --- CPU Implementation of Bitonic Sort ---
void bitonicSortCPU(std::vector<int>& arr) {
	int n = arr.size();

	// Внешний цикл: проход по размеру подпоследовательностей
	for (int k = 2; k <= n; k *= 2) {
		// Цикл для чередования направлений сортировки
		for (int j = k / 2; j > 0; j /= 2) {
			// Проход по всем элементам
			for (int i = 0; i < n; i++) {
				int ixj = i ^ j; // Индекс элемента для сравнения с текущим
				if (ixj > i) {
					// Определяем направление сортировки: на возрастание или убывание
					if ((i & k) == 0) {
						// Сортировка по возрастанию
						if (arr[i] > arr[ixj]) {
							std::swap(arr[i], arr[ixj]);
						}
					}
					else {
						// Сортировка по убыванию
						if (arr[i] < arr[ixj]) {
							std::swap(arr[i], arr[ixj]);
						}
					}
				}
			}
		}
	}
}


// Функция для замера времени выполнения на CPU
void measure_cpu(std::vector<int> src, std::vector<int>& result)
{

	auto start = std::chrono::high_resolution_clock::now();

	// Перестановка блоков
	bitonicSortCPU(src);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;

	std::cout << "\nCPU Time: " << duration.count() << " seconds" << std::endl;
	result = src;
}

// CUDA kernel для битонической сортировки
__global__ void bitonicSortCUDA(int* arr, int size, int step, int j) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int ixj = idx ^ j; // XOR для получения индекса элемента для сравнения

	if (ixj > idx && idx < size) {
		bool up = ((idx & step) == 0); // определение направления сортировки
		if ((arr[idx] > arr[ixj]) == up) {
			int temp = arr[idx];
			arr[idx] = arr[ixj];
			arr[ixj] = temp;
		}
	}
}

// Функция для запуска битонической сортировки на GPU
void bitonicSortOnGPU(int* d_arr, int size)
{
	int blockSize = 32 * 32;
	int numBlocks = (size + blockSize - 1) / blockSize;
	for (int k = 2; k <= size; k *= 2) {
		for (int j = k / 2; j > 0; j /= 2) {
			int numBlocks = (size + blockSize - 1) / blockSize;
			bitonicSortCUDA << <numBlocks, blockSize >> > (d_arr, size, k, j);
			cudaDeviceSynchronize(); // синхронизация всех блоков на каждом шаге
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
				return;
			}
		}
	}
}

static void measure_gpu_s(int* src, int size)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Запуск GPU-версии функции
	bitonicSortOnGPU(src, size);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "\nGPU Time: " << milliseconds / 1000.0 << " seconds" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Полное поэлементное сравнение массивов
bool compare_results(const std::vector<int>& cpu_arr, const std::vector<int>& gpu_arr)
{
	for (int i = 0; i < cpu_arr.size(); ++i)
		if (fabsf(cpu_arr[i] - gpu_arr[i]) > 1e-5)
			return false;
	return true;
}

void print_partial_array(std::vector<int> arr, int size)
{
	for (int i = 0; i < arr.size() && i < size; i++)
		std::cout << arr[i] << " ";
}

int main()
{

	// Создание массива на CPU
	auto src_array = create_random_array(N);
	std::vector<int> cpu_result(N), gpu_result(N);

	int* d_arr;
	size_t size_in_bytes = src_array.size() * sizeof(int);
	size_t free_mem;

	cudaMemGetInfo(&free_mem, 0);
	std::cout << "Free memory: " << free_mem / 1024 / 1024 << "MB" << std::endl;
	std::cout << "Memory need to allocate: " << size_in_bytes / 1024 / 1024 << "MB" << std::endl;

	if (size_in_bytes < free_mem)
	{
		auto error = cudaMalloc(&d_arr, size_in_bytes);
		if (error != cudaSuccess)
		{
			std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
			return; // или другой механизм обработки ошибки
		}
		else {
			std::cout << "Memory successfully allocated on device!" << std::endl;
		}
	}
	else
		return;

	std::cout << "Source:" << std::endl;
	print_partial_array(src_array, 8);
	std::cout << std::endl;

	// Замер времени на CPU
	measure_cpu(src_array, cpu_result);

	// Вывод части массива с CPU
	std::cout << "Partial array CPU:" << std::endl;
	print_partial_array(cpu_result, 8);
	std::cout << std::endl;

	cudaMemcpy(d_arr, src_array.data(), size_in_bytes, cudaMemcpyHostToDevice);

	// Замер времени на GPU
	measure_gpu_s(d_arr, src_array.size());

	// Получение результатов с GPU
	cudaMemcpy(gpu_result.data(), d_arr, size_in_bytes, cudaMemcpyDeviceToHost);

	std::cout << "Partial array GPU with shared memory:" << std::endl;
	print_partial_array(gpu_result, 8);
	std::cout << std::endl;

	// Сравнение результатов
	if (compare_results(cpu_result, gpu_result)) {
		std::cout << "\nCPU and GPU results match!" << std::endl;
	}
	else {
		std::cout << "\nResults differ!" << std::endl;
	}

	// Освобождение памяти
	cudaFree(d_arr);

	return 0;
}
