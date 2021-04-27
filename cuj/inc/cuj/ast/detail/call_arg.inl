#pragma once

#include <cuj/ast/context.h>
#include <cuj/ir/builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace call_detail
{

    template<typename T>
    void prepare_arg(
        ir::IRBuilder               &builder,
        const Value<T>              &value,
        std::vector<ir::BasicValue> &output)
    {
        if constexpr(is_cuj_class<T> || is_array<T>)
        {
            output.push_back(value.address().get_impl()->gen_ir(builder));
        }
        else
        {
            output.push_back(value.get_impl()->gen_ir(builder));
        }
    }

} // namespace call_detail

CUJ_NAMESPACE_END(cuj::ast)
