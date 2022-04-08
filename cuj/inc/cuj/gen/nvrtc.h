#pragma once

#ifdef CUJ_ENABLE_CUDA

#include <cuj/dsl/module.h>
#include <cuj/gen/option.h>
#include <cuj/utils/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NVRTC : public Uncopyable
{
public:

    void set_options(const Options &opts);

    void generate(const dsl::Module &mod);

    const std::string &get_ptx() const;

    const std::string &get_log() const;

private:

    Options     opts_;
    std::string ptx_;
    std::string log_;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #ifdef CUJ_ENABLE_CUDA
