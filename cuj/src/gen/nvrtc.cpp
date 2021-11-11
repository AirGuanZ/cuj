#ifdef CUJ_ENABLE_CUDA

#include <array>

#include <nvrtc.h>

#include <cuj/gen/c.h>
#include <cuj/gen/nvrtc.h>
#include <cuj/util/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{
    void check_nvrtc_error(nvrtcResult result)
    {
        if(result != NVRTC_SUCCESS)
            throw CUJException(nvrtcGetErrorString(result));
    }
}

void NVRTC::generate(const ir::Program &prog)
{
    generate(prog, { true, false });
}

void NVRTC::generate(const ir::Program &prog, const Options &opts)
{
    std::string c_src;
    {
        CGenerator c_generator;
        c_generator.set_cuda();
        c_generator.print(prog);
        c_src = c_generator.get_string();
    }

    nvrtcProgram program = nullptr;
    check_nvrtc_error(nvrtcCreateProgram(
        &program, c_src.data(), nullptr, 0, nullptr, nullptr));
    CUJ_SCOPE_GUARD({ nvrtcDestroyProgram(&program); });

    int option_count = 0;
    std::array<const char *, 4> options;

    options[option_count++] = "--std=c++17";
    options[option_count++] = "--extra-device-vectorization";

    if(opts.reloc)
        options[option_count++] = "-rdc=true";
    
    if(opts.fast_math)
        options[option_count++] = "--use_fast_math";

    const auto result = nvrtcCompileProgram(
        program, option_count, options.data());
    if(result != NVRTC_SUCCESS)
    {
        size_t log_size;
        check_nvrtc_error(nvrtcGetProgramLogSize(program, &log_size));

        std::vector<char> log(log_size);
        check_nvrtc_error(nvrtcGetProgramLog(program, log.data()));

        throw CUJException(log.data());
    }

    {

        size_t log_size;
        check_nvrtc_error(nvrtcGetProgramLogSize(program, &log_size));

        std::vector<char> log(log_size);
        check_nvrtc_error(nvrtcGetProgramLog(program, log.data()));

        printf("%s\n", log.data());
    }

    size_t ptx_size;
    check_nvrtc_error(nvrtcGetPTXSize(program, &ptx_size));

    std::vector<char> ptx_data(ptx_size);
    check_nvrtc_error(nvrtcGetPTX(program, ptx_data.data()));

    ptx_ = ptx_data.data();
}

const std::string &NVRTC::get_ptx() const
{
    return ptx_;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #ifdef CUJ_ENABLE_CUDA
