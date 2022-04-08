#ifdef CUJ_ENABLE_CUDA

#include <array>

#include <nvrtc.h>

#include <cuj/gen/cpp.h>
#include <cuj/gen/nvrtc.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{
    void check_nvrtc_error(nvrtcResult result)
    {
        if(result != NVRTC_SUCCESS)
            throw CujException(nvrtcGetErrorString(result));
    }
}

void NVRTC::set_options(const Options &opts)
{
    opts_ = opts;
}

void NVRTC::generate(const dsl::Module &mod)
{
    std::string c_src;
    {
        CPPCodeGenerator c_generator;
        c_generator.set_target(CPPCodeGenerator::Target::PTX);
        c_generator.set_assert(opts_.enable_assert);
        c_generator.generate(mod);
        c_src = c_generator.get_cpp_string();
    }

    nvrtcProgram program = nullptr;
    check_nvrtc_error(nvrtcCreateProgram(
        &program, c_src.data(), nullptr, 0, nullptr, nullptr));
    CUJ_SCOPE_EXIT{ nvrtcDestroyProgram(&program); };

    int option_count = 0;
    std::array<const char *, 4> options;

    options[option_count++] = "--std=c++17";
    options[option_count++] = "--extra-device-vectorization";
    options[option_count++] = "-rdc=true";
    if(opts_.fast_math)
        options[option_count++] = "--use_fast_math";

    const auto result = nvrtcCompileProgram(
        program, option_count, options.data());

    {

        size_t log_size;
        check_nvrtc_error(nvrtcGetProgramLogSize(program, &log_size));

        std::vector<char> log(log_size);
        check_nvrtc_error(nvrtcGetProgramLog(program, log.data()));

        log_ = log.data();
    }

    if(result != NVRTC_SUCCESS)
        throw CujException(log_);

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

const std::string &NVRTC::get_log() const
{
    return log_;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #ifdef CUJ_ENABLE_CUDA
