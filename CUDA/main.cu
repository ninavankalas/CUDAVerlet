#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <inttypes.h>
#include <cuda.h>
#include "cutil.h"

#define ITERATIONS 10
#define YEARS_PER_STEP 1000

#define Epsilon2 7.3890560989306495
#define GRAV_CONST 6.6738480e-11

struct Params {
	size_t count;
	uint32_t *hip;
	double3 *pos;
	double3 *vel;
	double *mass;
};

template<unsigned int blockSize>
__global__ void compute_accelerations(Params p, double3 * __restrict acc)
{
	extern __shared__ double3 s_sum[];

	uint32_t tid = threadIdx.x;
	uint32_t i = blockIdx.x * blockSize + threadIdx.x;

	s_sum[tid].x = 0;
	s_sum[tid].y = 0;
	s_sum[tid].z = 0;

	if (i >= (uint32_t) p.count)
		return;

	const size_t count = p.count;
	const double3 *a = &p.pos[i];

	for (size_t j = 0; j < count; j++) {
		const double3 *b = &p.pos[j];
		double3 r;
		r.x = b->x - a->x;
		r.y = b->y - a->y;
		r.z = b->z - a->z;
		double dist = r.x * r.x + r.y * r.y + r.z * r.z + Epsilon2;
		double s = p.mass[j] / (dist * sqrt(dist));
		s_sum[tid].x += r.x * s;
		s_sum[tid].y += r.y * s;
		s_sum[tid].z += r.z * s;
	}

	s_sum[tid].x *= GRAV_CONST;
	s_sum[tid].y *= GRAV_CONST;
	s_sum[tid].z *= GRAV_CONST;
	__syncthreads();

	// Reduction
	if (blockSize >= 1024) {
		if (tid < 512) {
			s_sum[tid].x += s_sum[tid + 512].x;
			s_sum[tid].y += s_sum[tid + 512].y;
			s_sum[tid].z += s_sum[tid + 512].z;
		}
		__syncthreads();
	}
	if (blockSize >= 512) {
		if (tid < 256) {
			s_sum[tid].x += s_sum[tid + 256].x;
			s_sum[tid].y += s_sum[tid + 256].y;
			s_sum[tid].z += s_sum[tid + 256].z;
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			s_sum[tid].x += s_sum[tid + 128].x;
			s_sum[tid].y += s_sum[tid + 128].y;
			s_sum[tid].z += s_sum[tid + 128].z;
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			s_sum[tid].x += s_sum[tid + 64].x;
			s_sum[tid].y += s_sum[tid + 64].y;
			s_sum[tid].z += s_sum[tid + 64].z;
		}
		__syncthreads();
	}
	if (tid < 32) {
		if (blockSize >= 64) {
			s_sum[tid].x += s_sum[tid + 32].x;
			s_sum[tid].y += s_sum[tid + 32].y;
			s_sum[tid].z += s_sum[tid + 32].z;
			__syncthreads();
		}
		if (blockSize >= 32) {
			s_sum[tid].x += s_sum[tid + 16].x;
			s_sum[tid].y += s_sum[tid + 16].y;
			s_sum[tid].z += s_sum[tid + 16].z;
			__syncthreads();
		}
		if (blockSize >= 16) {
			s_sum[tid].x += s_sum[tid + 8].x;
			s_sum[tid].y += s_sum[tid + 8].y;
			s_sum[tid].z += s_sum[tid + 8].z;
			__syncthreads();
		}
		if (blockSize >= 8) {
			s_sum[tid].x += s_sum[tid + 4].x;
			s_sum[tid].y += s_sum[tid + 4].y;
			s_sum[tid].z += s_sum[tid + 4].z;
			__syncthreads();
		}
		if (blockSize >= 4) {
			s_sum[tid].x += s_sum[tid + 2].x;
			s_sum[tid].y += s_sum[tid + 2].y;
			s_sum[tid].z += s_sum[tid + 2].z;
			__syncthreads();
		}
		if (blockSize >= 2) {
			s_sum[tid].x += s_sum[tid + 1].x;
			s_sum[tid].y += s_sum[tid + 1].y;
			s_sum[tid].z += s_sum[tid + 1].z;
			__syncthreads();
		}
	}
	if (tid == 0)
		acc[blockIdx.x] = s_sum[0];
}

#define dt YEARS_PER_STEP
#define hdt (dt / 2.)
#define sdt (dt * hdt)
template<unsigned int blockSize>
__global__ void move_stars(Params p, const double3 * __restrict acc,
		double3 * __restrict dpos)
{
	uint32_t i = blockIdx.x * blockSize + threadIdx.x;

	if (i >= (uint32_t) p.count)
		return;

	double dx = p.vel[i].x * dt + acc[i].x * sdt;
	double dy = p.vel[i].y * dt + acc[i].y * sdt;
	double dz = p.vel[i].z * dt + acc[i].z * sdt;
	p.pos[i].x += dx;
	p.pos[i].y += dy;
	p.pos[i].z += dz;
	p.vel[i].x = dpos[i].x / dt + acc[i].x * hdt;
	p.vel[i].y = dpos[i].y / dt + acc[i].y * hdt;
	p.vel[i].z = dpos[i].z / dt + acc[i].z * hdt;
	dpos[i].x = dx;
	dpos[i].y = dy;
	dpos[i].z = dz;
}
#undef sdt
#undef hdt
#undef dt

#define BLOCK_THREADS 512
__host__ void simulate(Params hostParam, cudaDeviceProp& prop)
{
	size_t vec_size = hostParam.count * sizeof(double3);
	size_t mass_size = hostParam.count * sizeof(double);
	Params devParam;

	size_t blocks = ceil(((double) hostParam.count) / BLOCK_THREADS);
	size_t sharedmem = BLOCK_THREADS * sizeof(double3);

	if (sharedmem > prop.sharedMemPerBlock) {
		printf("Error: there is not enough shared memory per block.");
		exit(EXIT_FAILURE);
	}

	double3 *devAcc, *devDpos;
	cudaSafeCall(cudaMalloc(&devAcc, vec_size));
	cudaSafeCall(cudaMalloc(&devDpos, vec_size));
	cudaSafeCall(cudaMalloc(&devParam.pos, vec_size));
	cudaSafeCall(cudaMalloc(&devParam.vel, vec_size));
	cudaSafeCall(cudaMalloc(&devParam.mass, mass_size));

	devParam.count = hostParam.count;
	cudaSafeCall(cudaMemcpy(devParam.pos, hostParam.pos, vec_size, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(devParam.vel, hostParam.vel, vec_size, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(devParam.mass, hostParam.mass, mass_size, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemset(devAcc, 0, vec_size));
	cudaSafeCall(cudaMemset(devDpos, 0, vec_size));

	dim3 dimBlock(blocks, 1, 1);
	dim3 dimThreads(BLOCK_THREADS, 1, 1);

	for (size_t it = 0; it < ITERATIONS; it++) {
		// Get acceleration of each star in relation with all other stars
		compute_accelerations<BLOCK_THREADS> <<<dimBlock, dimThreads, sharedmem>>>(devParam, devAcc);
		cudaCheckError();

		// Move all stars
		move_stars<BLOCK_THREADS> <<<dimBlock, dimThreads>>>(devParam, devAcc, devDpos);
		cudaCheckError();
	}

	cudaSafeCall(cudaMemcpy(hostParam.pos, devParam.pos, vec_size, cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaFree(devAcc));
	cudaSafeCall(cudaFree(devDpos));
	cudaSafeCall(cudaFree(devParam.pos));
	cudaSafeCall(cudaFree(devParam.vel));
	cudaSafeCall(cudaFree(devParam.mass));
}

__host__ void print_stars(Params p)
{
	printf("Ano %ld\n", ITERATIONS * YEARS_PER_STEP);
	for (size_t s = 0; s < p.count; s++) {
		double3 *pos = &p.pos[s];
		printf("%d %.5lf %.5lf %.5lf\n", p.hip[s], pos->x, pos->y, pos->z);
	}
}

int main(void)
{
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 3;

	cudaSafeCall(cudaChooseDevice(&dev, &prop));
	cudaSafeCall(cudaSetDevice(dev));
	cudaGetDeviceProperties(&prop, dev);

	struct Params p;

	int r = scanf("%zu", &p.count);

	if (r != 1) {
		printf("Error: invalid input.\n");
		exit(EXIT_FAILURE);
	}

	p.hip = new uint32_t[p.count];
	p.pos = new double3[p.count];
	p.vel = new double3[p.count];
	p.mass = new double[p.count];
	for (size_t i = 0; i < p.count; i++) {
		int r = scanf("%d %le %le %le %le %le %le %le", &p.hip[i], &p.pos[i].x,
				&p.pos[i].y, &p.pos[i].z, &p.vel[i].x, &p.vel[i].y,
				&p.vel[i].z, &p.mass[i]);
		if (r != 8) {
			printf("Error: invalid input.\n");
			exit(EXIT_FAILURE);
		}
	}
	
	simulate(p, prop);

	//print_stars(p);

	delete[] p.hip;
	delete[] p.pos;
	delete[] p.vel;
	delete[] p.mass;

	return EXIT_SUCCESS;
}
