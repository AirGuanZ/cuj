#pragma once

#include <catch2/catch.hpp>

#include <cuj.h>

#include "cuda/cuda.h"

using namespace cuj;

namespace test_detail
{

    template<typename Tuple, int...Is>
    auto tuple_prefix(const Tuple &tuple, std::integer_sequence<int, Is...>)
    {
        return std::tuple{ std::get<Is>(tuple)... };
    }

} // namespace test_detail

template<typename F, typename Action>
void with_mcjit(F &&f, Action &&action)
{
    ScopedModule mod;
    auto func = function(std::forward<F>(f));
    MCJIT mcjit;
    mcjit.generate(mod);
    auto c_func = mcjit.get_function(func);
    REQUIRE(c_func);
    if(c_func)
        std::forward<Action>(action)(c_func);
}

template<typename Ret, typename F, typename Action>
    requires dsl::is_cuj_var_v<Ret> || dsl::is_cuj_ref_v<Ret>
void with_mcjit(F &&f, Action &&action)
{
    ScopedModule mod;
    auto func = function<Ret>(std::forward<F>(f));
    MCJIT mcjit;
    mcjit.generate(mod);
    auto c_func = mcjit.get_function(func);
    REQUIRE(c_func);
    if(c_func)
        std::forward<Action>(action)(c_func);
}

template<typename F, typename...Args>
void mcjit_require(F &&f, Args...args)
{
    with_mcjit(std::forward<F>(f), [&](auto c_func)
    {
        auto func_args = test_detail::tuple_prefix(
            std::tuple{ args... },
            std::make_integer_sequence<int, sizeof...(args) - 1>());
        auto result = std::get<sizeof...(args) - 1>(std::tuple{ args... });
        if constexpr(std::is_floating_point_v<decltype(result)>)
        {
            REQUIRE(
                std::apply(c_func, func_args) == Approx(result));
        }
        else
        {
            REQUIRE(
                std::apply(c_func, func_args) == result);
        }
    });
}

#if CUJ_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

namespace test_detail
{

    inline void check_cuda_error(cudaError_t err)
    {
        if(err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }

    template<typename Ret, typename F, typename...Args>
    auto build_kernel(F &&f, const std::tuple<Args...> &)
    {
        auto entry = [&](ptr<cxx<Ret>> ret, cxx<Args>...args)
        {
            *ret = std::apply(
                std::forward<F>(f), std::tuple<cxx<Args>...>{ args... });
        };
        return kernel("entry", entry);
    }

} // namespace test_detail

template<typename F, typename...Args>
void cuda_require(F &&f, Args...args)
{
    auto func_args = test_detail::tuple_prefix(
        std::tuple{ args... },
        std::make_integer_sequence<int, sizeof...(args) - 1>());
    auto result = std::get<sizeof...(args) - 1>(std::tuple{ args... });
    using Ret = decltype(result);

    ScopedModule mod;
    auto func = test_detail::build_kernel<Ret>(std::forward<F>(f), func_args);
    PTXGenerator ptx_gen;
    ptx_gen.generate(mod);
    const auto &ptx = ptx_gen.get_ptx();

    CUdevice cuda_device;
    CUcontext cuda_context;
    cuDeviceGet(&cuda_device, 0);
    cuCtxCreate(&cuda_context, 0, cuda_device);
    CUJ_SCOPE_EXIT{ cuCtxDestroy(cuda_context); };

    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

    Ret *device_ret = nullptr;
    test_detail::check_cuda_error(cudaMalloc(&device_ret, sizeof(result)));
    CUJ_SCOPE_EXIT{ cudaFree(device_ret); };

    auto launch = [&]<typename...Args_>(Args_...args_)
    {
        cuda_module.launch(
            "entry", { 1, 1, 1 }, { 1, 1, 1 },
            device_ret, args_...);
    };
    std::apply(launch, func_args);

    Ret computed_result;
    test_detail::check_cuda_error(cudaMemcpy(
        &computed_result, device_ret, sizeof(result), cudaMemcpyDeviceToHost));

    if constexpr(std::is_floating_point_v<Ret>)
        REQUIRE(computed_result == Approx(result));
    else
        REQUIRE(computed_result == result);
}

#endif // #if CUJ_ENABLE_CUDA
