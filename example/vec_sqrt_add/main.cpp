#include <iomanip>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuj/cuj.h>

using namespace cuj::builtin;
using namespace cuj::ast;

void check_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

std::string generate_ptx()
{
    Context context;
    CUJ_SCOPED_CONTEXT(&context);

    context.begin_function<void(float*,float*,float*,int)>(
        "vec_add", cuj::ir::Function::Type::Kernel);
    
    $arg(float*, A);
    $arg(float*, B);
    $arg(float*, C);
    $arg(int,    N);

    $var(int, i);

    i = cuda::thread_index_x() + cuda::block_index_x() * cuda::block_dim_x();

    $if(i < N)
    {
        C[i] = math::sqrt(A[i] + B[i]);
    };

    context.end_function();

    std::cout << "=========== cujitter ir ===========" << std::endl << std::endl;

    auto ir = context.gen_ir();
    cuj::gen::IRPrinter printer;
    printer.print(ir);
    std::cout << printer.get_string() << std::endl;

    std::cout << "=========== llvm ir ===========" << std::endl << std::endl;

    cuj::gen::LLVMIRGenerator llvm_gen;
    llvm_gen.set_target(cuj::gen::LLVMIRGenerator::Target::PTX);
    llvm_gen.generate(ir);
    std::cout << llvm_gen.get_string() << std::endl;

    std::cout << "=========== ptx ===========" << std::endl << std::endl;

    auto ptx = context.gen_ptx();
    std::cout << ptx << std::endl;

    return ptx;
}

void test_ptx(const std::string &ptx)
{
    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    CUJ_SCOPE_GUARD({ cuCtxDestroy(context); });

    cuj::CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

    constexpr int N = 10;

    std::vector<float> data_A, data_B, data_C;
    for(int j = 0; j < N; ++j)
    {
        data_A.push_back(j);
        data_B.push_back(2 * j);
        data_C.push_back(0);
    }

    float *device_A = nullptr, *device_B = nullptr, *device_C = nullptr;

    check_cuda_error(
        cudaMalloc(reinterpret_cast<void **>(&device_A), sizeof(float) * N));
    CUJ_SCOPE_GUARD({ if(device_A) cudaFree(device_A); });

    check_cuda_error(
        cudaMalloc(reinterpret_cast<void **>(&device_B), sizeof(float) * N));
    CUJ_SCOPE_GUARD({ if(device_B) cudaFree(device_B); });

    check_cuda_error(
        cudaMalloc(reinterpret_cast<void **>(&device_C), sizeof(float) * N));
    CUJ_SCOPE_GUARD({ if(device_C) cudaFree(device_C); });

    check_cuda_error(cudaMemcpy(
        device_A, data_A.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(
        device_B, data_B.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    const int blockSize  = 64;
    const int blockCount = (N + blockSize - 1) / blockSize;

    int n = N;
    cuda_module.launch(
        "vec_add", { blockCount }, { blockSize },
        device_A, device_B, device_C, n);

    check_cuda_error(cudaMemcpy(
        data_C.data(), device_C, sizeof(float) * N, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaDeviceSynchronize());
    
    std::cout << "C[i] <- Sqrt(A[i] + B[i])" << std::endl;

    std::cout << "A:          ";
    for(float a : data_A) std::cout << std::setw(7) << a << " ";
    std::cout << std::endl;

    std::cout << "B:          ";
    for(float b : data_B) std::cout << std::setw(7) << b << " ";
    std::cout << std::endl;

    std::cout << "C:          ";
    for(float c : data_C) std::cout << std::setw(7) << c << " ";
    std::cout << std::endl;

    std::cout << "Expected C: ";
    for(int i = 0; i < N; ++i)
        std::cout << std::setw(7) << std::sqrtf(data_A[i] + data_B[i]) << " ";
    std::cout << std::endl;
}

int main()
{
    try
    {
        test_ptx(generate_ptx());
    }
    catch(const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
    }
}
