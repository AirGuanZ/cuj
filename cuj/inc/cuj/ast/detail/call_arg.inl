#pragma once

#include <cuj/ir/builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace call_detail
{

    template<typename T>
    void prepare_arg(
        ir::IRBuilder                         &builder,
        const RC<typename Value<T>::ImplType> &value,
        std::vector<ir::BasicValue>           &output)
    {
        if constexpr(is_cuj_class<T>)
        {
            output.push_back(value->get_address()->gen_ir(builder));
        }
        else if constexpr(is_array<T>)
        {
            output.push_back(value->data_ptr->arr_alloc->gen_ir(builder));
        }
        else
        {
            output.push_back(value->gen_ir(builder));
        }
    }

} // namespace call_detail

CUJ_NAMESPACE_END(cuj::ast)
