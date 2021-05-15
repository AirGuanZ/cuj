#include <test/test.h>

#if CUJ_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

TEST_CASE("string")
{
    SECTION("host")
    {
        ScopedContext ctx;

        auto test_helloworld = to_callable<bool>([]
        {
            auto s = "hello, world!"_cuj;

            boolean ret = true;
            ret = ret && (s[0] == 'h');
            ret = ret && (s[1] == 'e');
            ret = ret && (s[12] == '!');
            ret = ret && (s[13] == '\0');

            $return(ret);
        });

        auto test_strlen = to_callable<int>(
            [](const Pointer<char> &str)
        {
            $return(builtin::strlen(str));
        });

        auto test_strcmp = to_callable<int>(
            [](const Pointer<char> &a, const Pointer<char> &b)
        {
            $return(builtin::strcmp(a, b));
        });

        auto test_strcpy = to_callable<void>(
            [](const Pointer<char> &dst, const Pointer<char> &src)
        {
            builtin::strcpy(dst, src);
        });

        auto test_memcpy = to_callable<void>(
            [](const Pointer<void> &dst, const Pointer<void> &src, usize bytes)
        {
            builtin::memcpy(dst, src, bytes);
        });

        auto test_memset = to_callable<void>(
            [](const Pointer<void> &dst, i32 ch, usize bytes)
        {
            builtin::memset(dst, ch, bytes);
        });

        auto jit = ctx.gen_native_jit();

        auto test_helloworld_func = jit.get_function(test_helloworld);
        REQUIRE(test_helloworld_func);
        if(test_helloworld_func)
            REQUIRE(test_helloworld_func() == true);

        auto test_strlen_func = jit.get_function(test_strlen);
        REQUIRE(test_strlen_func);
        if(test_strlen_func)
        {
            REQUIRE(test_strlen_func("123456") == 6);
            REQUIRE(test_strlen_func("") == 0);
        }

        auto test_strcmp_func = jit.get_function(test_strcmp);
        REQUIRE(test_strcmp_func);
        if(test_strcmp_func)
        {
            REQUIRE(test_strcmp_func("abc", "abcde") == -1);
            REQUIRE(test_strcmp_func("abc", "abc") == 0);
            REQUIRE(test_strcmp_func("adc", "abc") == 1);
        }

        auto test_strcpy_func = jit.get_function(test_strcpy);
        REQUIRE(test_strcpy_func);
        if(test_strcpy_func)
        {
            char src[] = "abcde";
            char dst[] = "12345";
            test_strcpy_func(dst, src);
            REQUIRE(std::strcmp(dst, "abcde") == 0);
        }

        auto test_memcpy_func = jit.get_function(test_memcpy);
        REQUIRE(test_memcpy_func);
        if(test_memcpy_func)
        {
            char src[] = "abcde";
            char dst[] = "12345";
            test_memcpy_func(dst, src, 4);
            REQUIRE(std::strcmp(dst, "abcd5") == 0);
        }

        auto test_memset_func = jit.get_function(test_memset);
        REQUIRE(test_memset_func);
        if(test_memset_func)
        {
            char dst[] = "12345";
            test_memset_func(dst + 1, 'a', 3);
            REQUIRE(std::strcmp(dst, "1aaa5") == 0);
        }
    }

#if CUJ_ENABLE_CUDA
    SECTION("ptx")
    {
        ScopedContext ctx;

        to_callable<void>(
            "test", ir::Function::Type::Kernel,
            [](const Pointer<char> &output)
        {
            i32 i = cuda::block_dim_x() * cuda::block_index_x()
                  + cuda::thread_index_x();

            auto s = "hello, world!"_cuj;

            $if(i <= 14)
            {
                output[i] = s[i];
            };
        });

        auto ptx = ctx.gen_ptx();

        CUdevice cuDevice;
        CUcontext context;
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&context, 0, cuDevice);
        CUJ_SCOPE_GUARD({ cuCtxDestroy(context); });

        CUDAModule cuda_module;
        cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

        char *device_output = nullptr;
        cudaMalloc(reinterpret_cast<void **>(&device_output), 14);
        CUJ_SCOPE_GUARD({ if(device_output) cudaFree(device_output); });

        const int blockSize = 64;
        const int blockCount = (20 + blockSize - 1) / blockSize;

        cuda_module.launch("test", { blockCount }, { blockSize }, device_output);

        std::vector<char> host_output(14);
        cudaMemcpy(host_output.data(), device_output, 14, cudaMemcpyDeviceToHost);

        REQUIRE(std::string(host_output.data()) == "hello, world!");
    }
#endif // #if CUJ_ENABLE_CUDA
}
