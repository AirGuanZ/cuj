#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

void store_f32x4(ptr<f32> addr, f32 a, f32 b, f32 c, f32 d);
void store_u32x4(ptr<u32> addr, u32 a, u32 b, u32 c, u32 d);
void store_i32x4(ptr<i32> addr, i32 a, i32 b, i32 c, i32 d);

void store_f32x2(ptr<f32> addr, f32 a, f32 b);
void store_u32x2(ptr<u32> addr, u32 a, u32 b);
void store_i32x2(ptr<i32> addr, i32 a, i32 b);

void load_f32x4(ptr<f32> addr, ref<f32> a, ref<f32> b, ref<f32> c, ref<f32> d);
void load_u32x4(ptr<u32> addr, ref<u32> a, ref<u32> b, ref<u32> c, ref<u32> d);
void load_i32x4(ptr<i32> addr, ref<i32> a, ref<i32> b, ref<i32> c, ref<i32> d);

void load_f32x2(ptr<f32> addr, ref<f32> a, ref<f32> b);
void load_u32x2(ptr<u32> addr, ref<u32> a, ref<u32> b);
void load_i32x2(ptr<i32> addr, ref<i32> a, ref<i32> b);

CUJ_NAMESPACE_END(cuj::cstd)
