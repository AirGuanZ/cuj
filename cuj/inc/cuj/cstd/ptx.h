#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

i32 thread_idx_x();
i32 thread_idx_y();
i32 thread_idx_z();

i32 block_idx_x();
i32 block_idx_y();
i32 block_idx_z();

i32 block_dim_x();
i32 block_dim_y();
i32 block_dim_z();

void sample_texture2d_1f(u64 texture_object, f32 u, f32 v, ref<f32> r);
void sample_texture2d_3f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b);
void sample_texture2d_4f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b, ref<f32> a);

void sample_texture2d_1i(u64 texture_object, f32 u, f32 v, ref<i32> r);
void sample_texture2d_3i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b);
void sample_texture2d_4i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b, ref<i32> a);

CUJ_NAMESPACE_END(cuj::cstd)
