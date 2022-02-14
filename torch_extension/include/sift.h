#pragma once
#include <torch/extension.h>

at::Tensor cube_select(at::Tensor points, const float radius);
at::Tensor cube_select_two(at::Tensor points, const float radius);
at::Tensor cube_select_four(at::Tensor points, const float radius);
at::Tensor multi_directional_knn(at::Tensor points, const float radius, const int m);