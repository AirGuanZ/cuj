#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>
#include <type_traits>

#define CUJ_NAMESPACE_BEGIN(NAME) namespace NAME {
#define CUJ_NAMESPACE_END(NAME)   }

#define CUJ_INTERNAL_ASSERT(...) assert(__VA_ARGS__)

CUJ_NAMESPACE_BEGIN(cuj)

template<typename T>
using rm_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template<typename T>
using RC = std::shared_ptr<T>;

template<typename T>
using Box = std::unique_ptr<T>;

template<typename T, typename...Args>
RC<T> newRC(Args &&...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T, typename...Args>
Box<T> newBox(Args &&...args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

struct UninitializeFlag { };

constexpr inline UninitializeFlag UNINIT;

[[noreturn]] inline void unreachable()
{
    std::terminate();
}

class CUJException : std::runtime_error
{
public:

    using runtime_error::runtime_error;
};

CUJ_NAMESPACE_END(cuj)
