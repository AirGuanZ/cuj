#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class PTXGenerator
{
public:

    struct Options
    {
        OptLevel opt_level  = OptLevel::Default;
        bool     fast_math  = false;
        bool     enable_slp = true;
    };
    
    void generate(const ir::Program &prog, OptLevel opt = OptLevel::Default);

    void generate(const ir::Program &prog, const Options &opts);

    const std::string &get_result() const;

private:
    
    std::string result_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
