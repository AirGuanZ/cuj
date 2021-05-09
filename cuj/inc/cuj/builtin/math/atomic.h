#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::atomic)

f32 atomic_add(const Pointer<f32> &dst, const f32 &val);
f64 atomic_add(const Pointer<f64> &dst, const f64 &val);

CUJ_NAMESPACE_END(cuj::builtin::atomic)
