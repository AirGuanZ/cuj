#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

i32 atomic_add(ptr<i32> dst, i32 val);

u32 atomic_add(ptr<u32> dst, u32 val);

f32 atomic_add(ptr<f32> dst, f32 val);

CUJ_NAMESPACE_END(cuj::cstd)
