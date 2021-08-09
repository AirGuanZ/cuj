#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/builtin/math/math.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::cuda)

enum class IntrinsicValueType
{
    thread_index_x,
    thread_index_y,
    thread_index_z,
    block_index_x,
    block_index_y,
    block_index_z,
    block_dim_x,
    block_dim_y,
    block_dim_z
};

using Dim3 = math::Vec3i;

Value<int> thread_index_x();
Value<int> thread_index_y();
Value<int> thread_index_z();

Value<int> block_index_x();
Value<int> block_index_y();
Value<int> block_index_z();

Value<int> block_dim_x();
Value<int> block_dim_y();
Value<int> block_dim_z();

Dim3 thread_index();
Dim3 block_index();
Dim3 block_dim();

void sync_block_threads();

CUJ_NAMESPACE_END(cuj::builtin::cuda)

#endif // #if CUJ_ENABLE_CUDA
