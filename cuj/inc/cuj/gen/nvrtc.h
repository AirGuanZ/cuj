#pragma once

#ifdef CUJ_ENABLE_CUDA

#include <cuj/ir/prog.h>
#include <cuj/gen/option.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NVRTC : public Uncopyable
{
public:

    void generate(const ir::Program &prog, const Options &opts);

    const std::string &get_ptx() const;

    const std::string &get_log() const;

private:

    std::string ptx_;
    std::string log_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #ifdef CUJ_ENABLE_CUDA
