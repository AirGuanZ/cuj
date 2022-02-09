#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuj.h>

#include "../../test/test/cuda/cuda.h"

using namespace cuj;

constexpr int OUTPUT_WIDTH  = 512;
constexpr int OUTPUT_HEIGHT = 512;

constexpr int TEX_WIDTH  = 3;
constexpr int TEX_HEIGHT = 3;

void check_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

void check_cuda_error(CUresult result)
{
    if(result != CUDA_SUCCESS)
    {
        const char *str = nullptr;
        if(cuGetErrorString(result, &str) == CUDA_SUCCESS)
            throw std::runtime_error(str);
        throw std::runtime_error("an unknown cuda error occurred");
    }
}

std::string generate_ptx()
{
    ScopedModule cuj_module;
    
    kernel("entry", [&](u64 tex, ptr<f32> output)
    {
        var x = cstd::block_idx_x() * cstd::block_dim_x() + cstd::thread_idx_x();
        var y = cstd::block_idx_y() * cstd::block_dim_y() + cstd::thread_idx_y();
        $if(x < OUTPUT_WIDTH & y < OUTPUT_HEIGHT)
        {
            var u = (f32(x) + 0.5f) * (1.0f / OUTPUT_WIDTH)  * TEX_WIDTH;
            var v = (f32(y) + 0.5f) * (1.0f / OUTPUT_HEIGHT) * TEX_HEIGHT;

            f32 r, g, b, a;
            cstd::sample_texture2d_4f(tex, u, v, r, g, b, a);

            var out_idx = 4 * (y * OUTPUT_WIDTH + x);
            output[out_idx + 0] = r;
            output[out_idx + 1] = g;
            output[out_idx + 2] = b;
            output[out_idx + 3] = a;
        };
    });

    PTXGenerator gen;
    gen.generate(cuj_module);
    return gen.get_ptx();
}

void run()
{
    const std::string ptx = generate_ptx();
    std::cout << ptx << std::endl;

    cuInit(0);

    CUdevice cuda_device;
    check_cuda_error(cuDeviceGet(&cuda_device, 0));

    CUcontext cuda_context;
    check_cuda_error(cuCtxCreate(&cuda_context, 0, cuda_device));
    CUJ_SCOPE_EXIT{ cuCtxDestroy(cuda_context); };
    
    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

    cudaChannelFormatDesc channel_format_desc;
    channel_format_desc.x = 32;
    channel_format_desc.y = 32;
    channel_format_desc.z = 32;
    channel_format_desc.w = 32;
    channel_format_desc.f = cudaChannelFormatKindFloat;

    cudaArray_t tex_arr;
    check_cuda_error(cudaMallocArray(
        &tex_arr, &channel_format_desc, TEX_WIDTH, TEX_HEIGHT));
    CUJ_SCOPE_EXIT{ cudaFreeArray(tex_arr); };

    std::vector<float> tex_data =
    {
        1, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 1, 1,
        
        0, 1, 0, 1,
        0, 0, 1, 1,
        1, 0, 0, 1,
        
        0, 0, 1, 1,
        1, 0, 0, 1,
        0, 1, 0, 1,
    };
    assert(tex_data.size() == TEX_WIDTH * TEX_HEIGHT * 4);
    check_cuda_error(cudaMemcpy2DToArray(
        tex_arr, 0, 0, tex_data.data(), TEX_WIDTH * 4 * sizeof(float),
        TEX_WIDTH * 4 * sizeof(float), TEX_HEIGHT, cudaMemcpyHostToDevice));
    
    cudaResourceDesc tex_rsc_desc;
    tex_rsc_desc.resType         = cudaResourceTypeArray;
    tex_rsc_desc.res.array.array = tex_arr;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode     = cudaFilterModeLinear;
    tex_desc.readMode       = cudaReadModeElementType;

    cudaTextureObject_t tex;
    check_cuda_error(cudaCreateTextureObject(
        &tex, &tex_rsc_desc, &tex_desc, nullptr));
    CUJ_SCOPE_EXIT{ cudaDestroyTextureObject(tex); };

    float *device_output = nullptr;
    check_cuda_error(cudaMalloc(
        &device_output, sizeof(float) * 4 * OUTPUT_WIDTH * OUTPUT_HEIGHT));
    CUJ_SCOPE_EXIT{ cudaFree(device_output); };

    constexpr int BLOCK_SIZE = 16;
    constexpr int BLOCK_CNT_X = (OUTPUT_WIDTH  + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int BLOCK_CNT_Y = (OUTPUT_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_module.launch(
        "entry",
        { BLOCK_CNT_X, BLOCK_CNT_Y, 1 },
        { BLOCK_SIZE, BLOCK_SIZE, 1 },
        tex, device_output);

    check_cuda_error(cudaDeviceSynchronize());

    std::vector<float> color_buffer(OUTPUT_WIDTH * OUTPUT_HEIGHT * 4);
    check_cuda_error(cudaMemcpy(
        color_buffer.data(), device_output,
        sizeof(float) * color_buffer.size(),
        cudaMemcpyDeviceToHost));

    std::ofstream fout("output.ppm");
    if(!fout)
    {
        throw std::runtime_error(
            "failed to create output image: output.ppm");
    }

    fout << "P3\n"
         << OUTPUT_WIDTH << " "
         << OUTPUT_HEIGHT << std::endl
         << 255 << std::endl;
    for(int i = 0, j = 0; i < OUTPUT_WIDTH * OUTPUT_HEIGHT; ++i, j += 4)
    {
        const float rf = color_buffer[j];
        const float gf = color_buffer[j + 1];
        const float bf = color_buffer[j + 2];

        const int ri = std::min(255, static_cast<int>(rf * 255));
        const int gi = std::min(255, static_cast<int>(gf * 255));
        const int bi = std::min(255, static_cast<int>(bf * 255));

        fout << ri << " " << gi << " " << bi << " ";
    }

    fout.close();
    std::cout << "result written to output.ppm" << std::endl;
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
