#pragma once

#ifdef CUJ_ENABLE_CUDA

#include <cuj/ir/prog.h>
#include <cuj/gen/llvm.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NVRTC : public Uncopyable
{
public:

    struct Options
    {
        bool reloc     = false;
        bool fast_math = false;
    };

    void generate(const ir::Program &prog);

    void generate(const ir::Program &prog, const Options &opts);

    const std::string &get_ptx() const;

private:

    std::string ptx_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #ifdef CUJ_ENABLE_CUDA
