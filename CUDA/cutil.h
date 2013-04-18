#ifndef _CUDA_ERROR_H_
#define _CUDA_ERROR_H_

#include <cstdio>
#include <cstdlib>

#define cudaSafeCall(func) _cudaSafeCall(func, __FILE__, __LINE__)

extern inline void _cudaSafeCall(const cudaError_t ret, const char *file,
		const int line)
{
	if (ret != cudaSuccess) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i: %s (err=%d)\n",
				file, line, cudaGetErrorString(ret), ret);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define cudaCheckError() _cudaCheckError(__FILE__, __LINE__)

extern inline void _cudaCheckError(const char *file, const int line)
{
#ifndef NDEBUG
	cudaDeviceSynchronize();
#endif
	if (cudaPeekAtLastError() != cudaSuccess) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i: %s\n", file,
				line, cudaGetErrorString(cudaGetLastError()));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#endif /* _CUDA_ERROR_H_ */
