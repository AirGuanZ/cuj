#include <test/test.h>

#ifdef CUJ_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

namespace
{
    void check_cuda_error(cudaError_t err)
    {
        if(err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }
}

#endif

TEST_CASE("builtin.math.atomic")
{
    SECTION("f32")
    {
        ScopedContext ctx;

        auto test = to_callable<float>(
            []
        {
            f32 sum = 0;
            for(int i = 1; i < 5; ++i)
                atomic::atomic_add(sum.address(), i);
            $return(sum);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == Approx(10));
    }

    SECTION("f64")
    {
        ScopedContext ctx;

        auto test = to_callable<double>(
            []
        {
            f64 sum = 0;
            for(int i = 1; i < 5; ++i)
                atomic::atomic_add(sum.address(), i);
            $return(sum);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == Approx(10));
    }

#ifdef CUJ_ENABLE_CUDA

    SECTION("cuda.f32")
    {
        ScopedContext ctx;

        auto kernel = to_kernel([&](Pointer<float> p)
        {
            atomic::atomic_add(p, 1.0f);
        });

        auto ptx = ctx.gen_ptx();

        CUdevice cu_device;
        CUcontext cu_context;
        cuDeviceGet(&cu_device, 0);
        cuCtxCreate(&cu_context, 0, cu_device);
        CUJ_SCOPE_GUARD({ cuCtxDestroy(cu_context); });

        CUDAModule cu_module;
        cu_module.load_ptx_from_memory(ptx.data(), ptx.size());

        float *device_float;
        check_cuda_error(cudaMalloc((void **)&device_float, sizeof(float)));

        cu_module.launch(kernel, { 1, 1, 1 }, { 10, 10, 1 }, device_float);
        check_cuda_error(cudaDeviceSynchronize());

        float host_float;
        check_cuda_error(cudaMemcpy(
            &host_float, device_float, sizeof(float), cudaMemcpyDeviceToHost));

        REQUIRE(host_float == Approx(100));
    }

#endif
}
