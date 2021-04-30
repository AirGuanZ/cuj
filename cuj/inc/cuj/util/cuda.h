#pragma once

#if CUJ_ENABLE_CUDA

#include <string>

#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj)

class CUDAModule : public Uncopyable
{
public:

    struct Dim3 { int x = 1, y = 1, z = 1; };

    CUDAModule();

    ~CUDAModule();

    void load_ptx_from_memory(const void *data, size_t bytes);

    void load_ptx_from_file(const std::string &filename);
    
    template<typename...Args>
    void launch(
        const std::string &entry_name,
        const Dim3        &block_cnt,
        const Dim3        &block_size,
        Args           &...kernel_args);

private:

    void link();

    template<typename Arg0>
    void take_kernel_arg_ptrs(void **arg_ptrs, Arg0 &arg0) const;

    template<typename Arg0, typename Arg1, typename...Args>
    void take_kernel_arg_ptrs(
        void **arg_ptrs, Arg0 &arg0, Arg1 &arg1, Args &...args) const;

    void launch_impl(
        const std::string &entry_name,
        const Dim3        &block_cnt,
        const Dim3        &block_size,
        void             **kernel_args);

    struct Impl;

    Box<Impl> impl_;
};

template<typename ... Args>
void CUDAModule::launch(
    const std::string &entry_name,
    const Dim3        &block_cnt,
    const Dim3        &block_size,
    Args           &...kernel_args)
{
    if constexpr(sizeof...(kernel_args) > 0)
    {
        void *kernel_arg_ptrs[sizeof...(kernel_args)];
        this->take_kernel_arg_ptrs(kernel_arg_ptrs, kernel_args...);
        this->launch_impl(entry_name, block_cnt, block_size, kernel_arg_ptrs);
    }
    else
        this->launch_impl(entry_name, block_cnt, block_size, nullptr);
}

template<typename Arg0>
void CUDAModule::take_kernel_arg_ptrs(void **arg_ptrs, Arg0 &arg0) const
{
    *arg_ptrs = &arg0;
}

template<typename Arg0, typename Arg1, typename...Args>
void CUDAModule::take_kernel_arg_ptrs(
    void **arg_ptrs, Arg0 &arg0, Arg1 &arg1, Args &...args) const
{
    *arg_ptrs = &arg0;
    this->take_kernel_arg_ptrs(arg_ptrs + 1, arg1, args...);
}

CUJ_NAMESPACE_END(cuj)

#endif // #if CUJ_ENABLE_CUDA