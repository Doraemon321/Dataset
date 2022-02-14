#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

__global__ void cube_select_kernel(int b, int n, float radius, const float *points, int *idx)
{
	int batch_idx = blockIdx.x;
	points += batch_idx * n * 3;
	idx += batch_idx * n * 8;

	float temp_dist[8];
	float r2_dist = radius * radius;

	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		float x = points[i * 3];
		float y = points[i * 3 + 1];
		float z = points[i * 3 + 2];
		for (int j = 0; j < 8; j++)
		{
			temp_dist[j] = 1e8;
			idx[i * 8 + j] = i;
		}

		for (int j = 0; j < n; j++)
		{
			if (i == j) continue;
			float tx = points[j * 3];
			float ty = points[j * 3 + 1];
			float tz = points[j * 3 + 2];
			float dist = (x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz);
			if (dist > r2_dist) continue;
			int _x = (tx > x);
			int _y = (ty > y);
			int _z = (tz > z);
			int temp_idx = _x * 4 + _y * 2 + _z;
			if (dist < temp_dist[temp_idx])
			{
				idx[i * 8 + temp_idx] = j;
				temp_dist[temp_idx] = dist;
			}
		}
	}
}

__global__ void cube_select_two_kernel(int b, int n, float radius, const float *points, int *idx)
{
	int batch_idx = blockIdx.x;
	points += batch_idx * n * 3;
	idx += batch_idx * n * 16;
	float temp_dist[16];
	float r2_dist = radius * radius;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		float x = points[i * 3];
		float y = points[i * 3 + 1];
		float z = points[i * 3 + 2];

		for (int j = 0; j < 16; j++)
		{
			temp_dist[j] = r2_dist;
			idx[i * 16 + j] = i;
		}

		for (int j = 0; j < n; j++)
		{
			if (i == j) continue;
			float tx = points[j * 3];
			float ty = points[j * 3 + 1];
			float tz = points[j * 3 + 2];
			float dist = (x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz);
			if (dist > r2_dist) continue;
			int _x = (tx > x);
			int _y = (ty > y);
			int _z = (tz > z);
			int temp_idx = _x * 8 + _y * 4 + _z * 2;
			bool flag = false;

			for (int k = 0; k < 2; k++)
			{
				if (dist < temp_dist[temp_idx + k])
				{
					flag = true;
				}
				if (flag)
				{
					for (int kk = 1; kk >= k + 1; kk--)
					{
						idx[i * 16 + temp_idx + kk] = idx[i * 16 + temp_idx + kk - 1];
						temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
					}
					idx[i * 16 + temp_idx + k] = i;
					temp_dist[temp_idx + k] = dist;
					break;
				}
			}
		}
	}
}

__global__ void cube_select_four_kernel(int b, int n, float radius, const float *points, int *idx)
{
	int batch_idx = blockIdx.x;
	points += batch_idx * n * 3;
	idx += batch_idx * n * 32;
	float temp_dist[32];
	float r2_dist = radius * radius;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		float x = points[i * 3];
		float y = points[i * 3 + 1];
		float z = points[i * 3 + 2];

		for (int j = 0; j < 32; j++)
		{
			temp_dist[j] = r2_dist;
			idx[i * 32 + j] = i;
		}

		for (int j = 0; j < n; j++)
		{
			if (i == j) continue;
			float tx = points[j * 3];
			float ty = points[j * 3 + 1];
			float tz = points[j * 3 + 2];
			float dist = (x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz);
			if (dist > r2_dist) continue;
			int _x = (tx > x);
			int _y = (ty > y);
			int _z = (tz > z);
			int temp_idx = _x * 16 + _y * 8 + _z * 4;
			bool flag = false;

			for (int k = 0; k < 4; k++)
			{
				if (dist < temp_dist[temp_idx + k])
				{
					flag = true;
				}
				if (flag)
				{
					for (int kk = 3; kk >= k + 1; kk--)
					{
						idx[i * 32 + temp_idx + kk] = idx[i * 32 + temp_idx + kk - 1];
						temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
					}
					idx[i * 32 + temp_idx + k] = i;
					temp_dist[temp_idx + k] = dist;
					break;
				}
			}
		}
	}
}

__global__ void multi_directional_knn_kernel(int b, int n, float radius, int m, const float *points, int *idx)
{
	int batch_idx = blockIdx.x;
	points += batch_idx * n * 3;
	idx += batch_idx * n * m * 16;
	float temp_dist[48];
	float r2_dist = radius * radius;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		float x = points[i * 3];
		float y = points[i * 3 + 1];
		float z = points[i * 3 + 2];

		for (int j = 0; j < m * 16; j++)
		{
			temp_dist[j] = r2_dist;
			idx[i * m * 16 + j] = i;
		}

		for (int j = 0; j < n; j++)
		{
			if (i == j) continue;
			float tx = points[j * 3];
			float ty = points[j * 3 + 1];
			float tz = points[j * 3 + 2];
			float dist = (x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz);
			if (dist > r2_dist) continue;
			if (tx == x) tx += 1e-5;
			int _x = (tx > x);
			int _z = (tz > z);
			int _y;
			float _k = (ty - y) / (tx - x);
			if (_k < -1) _y = 0;
			else if (_k < 0) _y = 1;
			else if (_k < 1) _y = 2;
			else _y = 3;
			int temp_idx = _z * m * 8 + _x * m * 4 + _y * m;
			bool flag = false;

			for (int k = 0; k < m; k++)
			{
				if (dist < temp_dist[temp_idx + k])
				{
					flag = true;
				}
				if (flag)
				{
					for (int kk = m - 1; kk >= k + 1; kk--)
					{
						idx[i * m * 16 + temp_idx + kk] = idx[i * m * 16 + temp_idx + kk - 1];
						temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
					}
					idx[i * m * 16 + temp_idx + k] = i;
					temp_dist[temp_idx + k] = dist;
					break;
				}
			}
		}
	}
}

__global__ void multi_directional_knn_one_kernel(int b, int n, float radius, const float *points, int *idx)
{
	int batch_idx = blockIdx.x;
	points += batch_idx * n * 3;
	idx += batch_idx * n * 16;
	float temp_dist[16];
	float r2_dist = radius * radius;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		float x = points[i * 3];
		float y = points[i * 3 + 1];
		float z = points[i * 3 + 2];

		for (int j = 0; j < 16; j++)
		{
			temp_dist[j] = r2_dist;
			idx[i * 16 + j] = i;
		}

		for (int j = 0; j < n; j++)
		{
			if (i == j) continue;
			float tx = points[j * 3];
			float ty = points[j * 3 + 1];
			float tz = points[j * 3 + 2];
			float dist = (x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz);
			if (dist > r2_dist) continue;
			if (tx == x) tx += 1e-5;
			int _x = (tx > x);
			int _z = (tz > z);
			int _y;
			float _k = (ty - y) / (tx - x);
			if (_k < -1) _y = 0;
			else if (_k < 0) _y = 1;
			else if (_k < 1) _y = 2;
			else _y = 3;
			int temp_idx = _z * 8 + _x * 4 + _y;

			if (dist < temp_dist[temp_idx])
			{
				idx[i * 16 + temp_idx] = j;
				temp_dist[temp_idx] = dist;
			}
		}
	}
}

void cube_select_kernel_wrapper(int b, int n, float radius, const float *points, int *idx)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cube_select_kernel << <b, opt_n_threads(n), 0, stream >> > (b, n, radius, points, idx);
	CUDA_CHECK_ERRORS();
}
void cube_select_two_kernel_wrapper(int b, int n, float radius, const float *points, int *idx)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cube_select_two_kernel << <b, opt_n_threads(n), 0, stream >> > (b, n, radius, points, idx);
	CUDA_CHECK_ERRORS();
}
void cube_select_four_kernel_wrapper(int b, int n, float radius, const float *points, int *idx)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cube_select_four_kernel << <b, opt_n_threads(n), 0, stream >> > (b, n, radius, points, idx);
	CUDA_CHECK_ERRORS();
}

void multi_directional_knn_kernel_wrapper(int b, int n, float radius, int m, const float *points, int *idx)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	if (m == 1)
	{
		multi_directional_knn_one_kernel << <b, opt_n_threads(n), 0, stream >> > (b, n, radius, points, idx);
	}
	else
	{
		multi_directional_knn_kernel << <b, opt_n_threads(n), 0, stream >> > (b, n, radius, m, points, idx);
	}
	CUDA_CHECK_ERRORS();
}
