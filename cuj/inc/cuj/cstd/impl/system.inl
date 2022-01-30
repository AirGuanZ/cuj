#pragma once

CUJ_NAMESPACE_BEGIN(cuj::cstd)

template<typename...Args>
i32 print(const std::string &format_string, Args...args)
{
    static_assert(((
        dsl::is_cuj_arithmetic_v<
            dsl::remove_reference_t<dsl::remove_var_wrapper_t<Args>>> ||
        dsl::is_cuj_pointer_v<
            dsl::remove_reference_t<dsl::remove_var_wrapper_t<Args>>>) && ...));

    ptr<char_t> fmt_str_ptr = string_literial(format_string);
    core::CallFunc call = {
        .intrinsic = core::Intrinsic::print,
        .args = { newRC<core::Expr>(fmt_str_ptr._load()) }
    };
    ((call.args.push_back(newRC<core::Expr>(args._load()))), ...);

    return i32::_from_expr(core::Expr(std::move(call)));
}

inline void unreachable()
{
    auto func = dsl::FunctionContext::get_func_context();
    func->append_statement(core::CallFuncStat{
        .call_expr = core::CallFunc{
            .intrinsic = core::Intrinsic::unreachable
        }
    });
}

CUJ_NAMESPACE_END(cuj::cstd)
