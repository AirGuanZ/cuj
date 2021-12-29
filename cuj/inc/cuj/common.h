#pragma once

#include <memory>
#include <stdexcept>

#define CUJ_NAMESPACE_BEGIN(NAME) namespace NAME {
#define CUJ_NAMESPACE_END(NAME)   }

CUJ_NAMESPACE_BEGIN(cuj)

template<typename T>
using RC = std::shared_ptr<T>;

template<typename T>
using Box = std::unique_ptr<T>;

template<typename T, typename...Args>
auto newRC(Args &&...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T, typename...Args>
auto newBox(Args &&...args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

class CujException : public std::runtime_error
{
public:

    using runtime_error::runtime_error;
};

CUJ_NAMESPACE_END(cuj)
