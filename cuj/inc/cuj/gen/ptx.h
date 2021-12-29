#pragma once

#include <cuj/gen/option.h>
#include <cuj/dsl/module.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class PTXGenerator
{
public:

    void set_options(const Options &opts);

    void generate(const dsl::Module &mod);

    const std::string &get_llvm_ir() const;

    const std::string &get_ptx() const;

private:

    Options     opts_;
    std::string llvm_ir_;
    std::string ptx_;
};

CUJ_NAMESPACE_END(cuj::gen)
