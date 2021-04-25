#pragma once

#if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class PTXGenerator
{
public:

    enum class Target
    {
        PTX32,
        PTX64
    };

    void set_target(Target target);

    void generate(const ir::Program &prog);

    const std::string &get_result() const;

private:

    Target target_ = sizeof(void*) == 4 ? Target::PTX32 : Target::PTX64;

    std::string result_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM
