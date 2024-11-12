#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

const int N = 10240;

// Функция для создания матрицы с случайными значениями
std::vector<int> create_random_array(int size) {
	std::vector<int> arr(size);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-100, 100);

	for (int i = 0; i < size; ++i)
			arr[i] = dis(gen);

	return arr;
}

// --- CPU Implementation of Bitonic Sort ---
void bitonicMerge(std::vector<int>& arr, int low, int cnt, bool dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		for (int i = low; i < low + k; i++) {
			if ((arr[i] > arr[i + k]) == dir) {
				std::swap(arr[i], arr[i + k]);
			}
		}
		bitonicMerge(arr, low, k, dir);
		bitonicMerge(arr, low + k, k, dir);
	}
}

void bitonicSortRec(std::vector<int>& arr, int low, int cnt, bool dir) {
	if (cnt > 1) {
		int k = cnt / 2;

		bitonicSortRec(arr, low, k, true);
		bitonicSortRec(arr, low + k, k, false);

		bitonicMerge(arr, low, cnt, dir);
	}
}

void bitonicSortCPU(std::vector<int>& arr) {
	int n = arr.size();
	bitonicSortRec(arr, 0, n, true);
}

// Функция для замера времени выполнения на CPU
void measure_cpu(const std::vector<int>* src, std::vector<int>& result) {

	auto arr = *src;
	auto start = std::chrono::high_resolution_clock::now();

	// Перестановка блоков
	bitonicSortCPU(arr);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;

	std::cout << "CPU Time: " << duration.count() << " seconds" << std::endl;
}

// --- CUDA Implementation of Bitonic Sort ---
__device__ void bitonicMergeCUDA(std::vector<int>& arr, int low, int cnt, bool dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		for (int i = low; i < low + k; i++) {
			int temp = arr[i];
			if ((arr[i] > arr[i + k]) == dir) {
				arr[i] = arr[i + k];
				arr[i + k] = temp;
			}
		}
		bitonicMergeCUDA(arr, low, k, dir);
		bitonicMergeCUDA(arr, low + k, k, dir);
	}
}

__device__ void bitonicSortRecCUDA(std::vector<int>& arr, int low, int cnt, bool dir) {
	if (cnt > 1) {
		int k = cnt / 2;

		bitonicSortRecCUDA(arr, low, k, true);
		bitonicSortRecCUDA(arr, low + k, k, false);

		bitonicMergeCUDA(arr, low, cnt, dir);
	}
}

__global__ void bitonicSortCUDA(std::vector<int>& arr, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		bitonicSortRecCUDA(arr, 0, n, true);
	}
}

// Функция для замера времени выполнения на GPU
static void measure_gpu(int* src, int size)
{

	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Запуск GPU-версии функции
	bitonicSortCUDA << <numBlocks, blockSize >> > (src);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "GPU Time: " << milliseconds / 1000.0 << " seconds" << std::endl;

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
	for (int i = 0; i < arr.size() || i < size; i++)
		std::cout << arr[i] << " ";
}

int main() 
{

	// Создание массива на CPU
	auto src_array = create_random_array(N);
	std::vector<int> cpu_result(N), gpu_result(N);

	int* d_arr;
	int size = src_array.size() * sizeof(int);
	cudaMalloc(&d_arr, size);
	cudaMemcpy(d_arr, src_array.data(), size, cudaMemcpyHostToDevice);

	// Замер времени на CPU
	std::vector<std::vector<float>> cpu_result(N, std::vector<float>(N));
	measure_cpu(&src_array, cpu_result);

	// Замер времени на GPU
	measure_gpu(d_arr, size);

	// Получение результатов с GPU
	cudaMemcpy(gpu_result.data(), d_arr, size, cudaMemcpyDeviceToHost);

	// Сравнение результатов
	if (compare_results(cpu_result, gpu_result, N / 2, N * 2)) {
		std::cout << "CPU and GPU results match!" << std::endl;
	}
	else {
		std::cout << "Results differ!" << std::endl;
	}

	std::cout << "Source:" << std::endl;
	print_partial_array(src_array, 8, 8);


	// Вывод части массива с CPU
	std::cout << "Partial array CPU:" << std::endl;
	print_partial_array(cpu_result, 8);

	// Вывод части матрицы с GPU
	std::cout << "\n\nPartial array GPU:" << std::endl;
	print_partial_array(gpu_result, 8);

	// Освобождение памяти
	cudaFree(d_arr);

	return 0;
}
