#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuj/cuj.h>

using namespace cuj;
using namespace builtin::math;
using namespace builtin::cuda;

void check_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

void record_kernel()
{

}

std::string generate_ptx()
{
    ScopedContext ctx;

    const auto start_time = std::chrono::steady_clock::now();
    record_kernel();
    const auto end_record_time = std::chrono::steady_clock::now();

    gen::Options options;
    options.fast_math = true;
    auto ptx = ctx.gen_ptx(options);

    const auto end_compile_time = std::chrono::steady_clock::now();

    std::cout << "record time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_record_time - start_time).count()
              << "ms" << std::endl;
    
    std::cout << "compile time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_compile_time - end_record_time).count()
              << "ms" << std::endl;

    return ptx;
}

void run()
{
    std::string ptx = generate_ptx();
    
    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    CUJ_SCOPE_GUARD({ cuCtxDestroy(context); });

    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());
    
    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 8;

    constexpr int BLOCK_COUNT_X = (WIDTH  + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    constexpr int BLOCK_COUNT_Y = (HEIGHT + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;

    std::cout << "start rendering" << std::endl;
    const auto start_time = std::chrono::steady_clock::now();

    for(int i = 0; i < 400; ++i)
    {
        cuda_module.launch(
            "render",
            { BLOCK_COUNT_X, BLOCK_COUNT_Y, 1 },
            { BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 });
    }

    cudaDeviceSynchronize();

    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "render time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count()
              << "ms" << std::endl;
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
