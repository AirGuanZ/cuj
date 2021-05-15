#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::system)

void print(const Pointer<char> &msg);

Pointer<void> malloc(const ArithmeticValue<size_t> &bytes);

void free(const ast::PointerImpl<void> &ptr);

CUJ_NAMESPACE_END(cuj::builtin::system)
