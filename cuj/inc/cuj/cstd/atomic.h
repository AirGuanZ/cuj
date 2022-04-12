#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

i32 atomic_add(ptr<i32> dst, i32 val);

u32 atomic_add(ptr<u32> dst, u32 val);

f32 atomic_add(ptr<f32> dst, f32 val);

i32 atomic_cmpxchg(ptr<i32> addr, i32 cmp, i32 new_val);

u32 atomic_cmpxchg(ptr<u32> addr, u32 cmp, u32 new_val);

u64 atomic_cmpxchg(ptr<u64> addr, u64 cmp, u64 new_val);

CUJ_NAMESPACE_END(cuj::cstd)
