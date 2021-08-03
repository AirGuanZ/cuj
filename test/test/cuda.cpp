#if CUJ_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <test/test.h>

namespace
{
    void check_cuda_error(cudaError_t err)
    {
        if(err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }
}

TEST_CASE("builtin.cuda")
{
    SECTION("special registers")
    {
        ScopedContext context;

        auto kernel = to_kernel(
            [](Pointer<int> output_thread_idx, Pointer<int> output_block_idx)
        {
            i32 thread_idx = cuda::thread_index_x();
            i32 block_idx = cuda::block_index_x();
            i32 linear_idx = block_idx * cuda::block_dim_x() + thread_idx;
            output_thread_idx[linear_idx] = thread_idx;
            output_block_idx[linear_idx] = block_idx;
        });

        auto ptx = context.gen_ptx();

        CUdevice cu_device;
        CUcontext cu_context;
        cuDeviceGet(&cu_device, 0);
        cuCtxCreate(&cu_context, 0, cu_device);
        CUJ_SCOPE_GUARD({ cuCtxDestroy(cu_context); });

        CUDAModule cu_module;
        cu_module.load_ptx_from_memory(ptx.data(), ptx.size());

        constexpr int BLOCK_DIM = 64;
        constexpr int BLOCK_CNT = 6;
        constexpr int THREAD_CNT = BLOCK_DIM * BLOCK_CNT;

        float *output_thread_idx, *output_block_idx;

        check_cuda_error(cudaMalloc(&output_thread_idx, sizeof(int) * THREAD_CNT));
        CUJ_SCOPE_GUARD({ cudaFree(output_thread_idx); });

        check_cuda_error(cudaMalloc(&output_block_idx,  sizeof(int) * THREAD_CNT));
        CUJ_SCOPE_GUARD({ cudaFree(output_block_idx); });

        check_cuda_error(cudaMemset(
            output_thread_idx, 0, sizeof(int) * THREAD_CNT));
        check_cuda_error(cudaMemset(
            output_block_idx, 0, sizeof(int) * THREAD_CNT));

        cu_module.launch(
            kernel, { BLOCK_CNT }, { BLOCK_DIM },
            output_thread_idx, output_block_idx);
        check_cuda_error(cudaDeviceSynchronize());

        std::vector<int> host_output_thread_idx(THREAD_CNT);
        std::vector<int> host_output_block_idx(THREAD_CNT);

        check_cuda_error(cudaMemcpy(
            host_output_thread_idx.data(),
            output_thread_idx,
            sizeof(int) * THREAD_CNT,
            cudaMemcpyDeviceToHost));

        check_cuda_error(cudaMemcpy(
            host_output_block_idx.data(),
            output_block_idx,
            sizeof(int) * THREAD_CNT,
            cudaMemcpyDeviceToHost));

        for(int block_idx = 0; block_idx < BLOCK_CNT; ++block_idx)
        {
            for(int thread_idx = 0; thread_idx < BLOCK_DIM; ++thread_idx)
            {
                const int linear_idx = block_idx * BLOCK_DIM + thread_idx;
                REQUIRE(host_output_thread_idx[linear_idx] == thread_idx);
                REQUIRE(host_output_block_idx[linear_idx] == block_idx);
            }
        }
    }

    SECTION("print")
    {
        ScopedContext ctx;

        auto kernel = to_kernel([]
        {
            system::print("hello, cuda.system.print!\n"_cuj);
        });

        auto ptx = ctx.gen_ptx();

        CUdevice cu_device;
        CUcontext cu_context;
        cuDeviceGet(&cu_device, 0);
        cuCtxCreate(&cu_context, 0, cu_device);
        CUJ_SCOPE_GUARD({ cuCtxDestroy(cu_context); });

        CUDAModule cu_module;
        cu_module.load_ptx_from_memory(ptx.data(), ptx.size());

        cu_module.launch(kernel, { 2 }, { 1 });

        check_cuda_error(cudaDeviceSynchronize());
    }
}

#endif // #if CUJ_ENABLE_CUDA
