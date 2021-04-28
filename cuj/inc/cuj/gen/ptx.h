#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class PTXGenerator
{
public:
    
    void generate(const ir::Program &prog);

    const std::string &get_result() const;

private:
    
    std::string result_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
