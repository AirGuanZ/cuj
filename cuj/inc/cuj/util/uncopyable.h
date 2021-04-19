#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

class Uncopyable
{
public:

    Uncopyable() = default;

    Uncopyable(const Uncopyable & )          = delete;
    Uncopyable(      Uncopyable &&) noexcept = default;

    Uncopyable &operator=(const Uncopyable & )          = delete;
    Uncopyable &operator=(      Uncopyable &&) noexcept = default;
};

CUJ_NAMESPACE_END(cuj)
