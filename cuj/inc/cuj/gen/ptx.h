#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/gen/option.h>
#include <cuj/ir/prog.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class PTXGenerator
{
public:

    void generate(const ir::Program &prog, const Options &opts);

    const std::string &get_result() const;

private:
    
    std::string result_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
