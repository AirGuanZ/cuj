#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin)

i32 strlen(Pointer<char> str);

i32 strcmp(Pointer<char> a, Pointer<char> b);

void strcpy(Pointer<char> dst, Pointer<char> src);

void memcpy(Pointer<void> dst, Pointer<void> src, usize bytes);

void memset(Pointer<void> dst, i32 ch, usize bytes);

CUJ_NAMESPACE_END(cuj::builtin)
