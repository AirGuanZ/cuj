#if CUJ_ENABLE_CUDA

#include <fstream>
#include <vector>

#include <cuda.h>

#include <cuj/util/cuda.h>
#include <cuj/util/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj)

namespace
{

    void check_cu(CUresult result)
    {
        if(result != CUDA_SUCCESS)
        {
            const char *err_msg;
            cuGetErrorString(result, &err_msg);
            throw CUJException(err_msg);
        }
    }

} // namespace anonymous

struct CUDAModule::Impl
{
    std::vector<std::string> ptx_data;

    CUmodule cu_module = nullptr;
};

CUDAModule::CUDAModule()
{
    impl_ = newBox<Impl>();
}

CUDAModule::CUDAModule(CUDAModule &&other) noexcept
{
    std::swap(impl_, other.impl_);
}

CUDAModule &CUDAModule::operator=(CUDAModule &&other) noexcept
{
    std::swap(impl_, other.impl_);
    return *this;
}

CUDAModule::~CUDAModule()
{
    if(impl_->cu_module)
        cuModuleUnload(impl_->cu_module);
}

void CUDAModule::load_ptx_from_memory(const void *data, size_t bytes)
{
    std::string new_data;
    new_data.resize(bytes);
    std::memcpy(new_data.data(), data, bytes);
    impl_->ptx_data.push_back(new_data);
}

void CUDAModule::load_ptx_from_file(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if(!fin)
        throw CUJException("failed to open file: " + filename);

    fin.seekg(0, std::ios::end);
    const auto len = fin.tellg();
    fin.seekg(0, std::ios::beg);

    if(!len)
        throw CUJException("empty ptx file: " + filename);

    std::vector<char> data(static_cast<size_t>(len) + 1, '\0');
    fin.read(data.data(), len);
    if(!fin)
        throw CUJException("failed to read context from " + filename);

    load_ptx_from_memory(data.data(), data.size());
}

void CUDAModule::link()
{
    CUJ_INTERNAL_ASSERT(!impl_->cu_module && !impl_->ptx_data.empty());

    CUlinkState link_state = nullptr;
    CUJ_SCOPE_GUARD({ if(link_state) cuLinkDestroy(link_state); });

    if(impl_->ptx_data.size() == 1)
    {
        check_cu(cuModuleLoadDataEx(
            &impl_->cu_module, impl_->ptx_data[0].data(),
            0, nullptr, nullptr));
    }
    else
    {
        check_cu(cuLinkCreate(0, nullptr, nullptr, &link_state));

        for(auto &ptx : impl_->ptx_data)
        {
            std::vector<char> data(
                ptx.data(), ptx.data() + ptx.size() + 1);

            check_cu(cuLinkAddData(
                link_state, CU_JIT_INPUT_PTX,
                data.data(), ptx.size(),
                nullptr, 0, nullptr, nullptr));
        }

        void *cubin = nullptr; size_t cubin_size = 0;
        check_cu(cuLinkComplete(link_state, &cubin, &cubin_size));

        check_cu(cuModuleLoadDataEx(
            &impl_->cu_module, cubin, 0, nullptr, nullptr));
    }
}

void CUDAModule::launch_impl(
    const std::string &entry_name,
    const Dim3        &block_cnt,
    const Dim3        &block_size,
    void             **kernel_args)
{
    if(!impl_->cu_module)
        link();
    CUJ_INTERNAL_ASSERT(impl_->cu_module);

    CUfunction entry_func = nullptr;
    check_cu(cuModuleGetFunction(
        &entry_func, impl_->cu_module, entry_name.c_str()));
    CUJ_INTERNAL_ASSERT(entry_func);

    check_cu(cuLaunchKernel(
        entry_func,
        block_cnt.x, block_cnt.y, block_cnt.z,
        block_size.x, block_size.y, block_size.z,
        0, nullptr, kernel_args, nullptr));
}

CUJ_NAMESPACE_END(cuj)

#endif // #if CUJ_ENABLE_CUDA
