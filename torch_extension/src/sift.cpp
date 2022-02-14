#include "sift.h"
#include "utils.h"

void cube_select_kernel_wrapper(int b, int n, float radius, const float *points, int *idx);
void cube_select_two_kernel_wrapper(int b, int n, float radius, const float *points, int *idx);
void cube_select_four_kernel_wrapper(int b, int n, float radius, const float *points, int *idx);
void multi_directional_knn_kernel_wrapper(int b, int n, float radius, int m, const float *points, int *idx);

at::Tensor cube_select(at::Tensor points, const float radius)
{
	CHECK_CONTIGUOUS(points);
	CHECK_IS_FLOAT(points);

	at::Tensor output = torch::zeros({ points.size(0), points.size(1), 8 }, at::device(points.device()).dtype(at::ScalarType::Int));

	if (points.is_cuda())
	{
		cube_select_kernel_wrapper(points.size(0), points.size(1), radius, points.data_ptr<float>(), output.data_ptr<int>());
	}
	else
	{
		AT_ASSERT(false, "CPU not supported");
	}

	return output;
}

at::Tensor cube_select_two(at::Tensor points, const float radius)
{
	CHECK_CONTIGUOUS(points);
	CHECK_IS_FLOAT(points);

	at::Tensor output = torch::zeros({ points.size(0), points.size(1), 16 }, at::device(points.device()).dtype(at::ScalarType::Int));

	if (points.is_cuda())
	{
		cube_select_two_kernel_wrapper(points.size(0), points.size(1), radius, points.data_ptr<float>(), output.data_ptr<int>());
	}
	else
	{
		AT_ASSERT(false, "CPU not supported");
	}

	return output;
}

at::Tensor cube_select_four(at::Tensor points, const float radius)
{
	CHECK_CONTIGUOUS(points);
	CHECK_IS_FLOAT(points);

	at::Tensor output = torch::zeros({ points.size(0), points.size(1), 32 }, at::device(points.device()).dtype(at::ScalarType::Int));

	if (points.is_cuda())
	{
		cube_select_four_kernel_wrapper(points.size(0), points.size(1), radius, points.data_ptr<float>(), output.data_ptr<int>());
	}
	else
	{
		AT_ASSERT(false, "CPU not supported");
	}

	return output;
}

at::Tensor multi_directional_knn(at::Tensor points, const float radius, const int m)
{
	CHECK_CONTIGUOUS(points);
	CHECK_IS_FLOAT(points);

	at::Tensor output = torch::zeros({ points.size(0), points.size(1), m * 16 }, at::device(points.device()).dtype(at::ScalarType::Int));

	if (points.is_cuda())
	{
		multi_directional_knn_kernel_wrapper(points.size(0), points.size(1), radius, m, points.data_ptr<float>(), output.data_ptr<int>());
	}
	else
	{
		AT_ASSERT(false, "CPU not supported");
	}

	return output;
}