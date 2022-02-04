#pragma once

CUJ_NAMESPACE_BEGIN(cuj::cstd)

namespace cstd_detail
{

    template<typename Arg>
    auto convert_print_arg(Arg arg)
    {
        using TArg = dsl::remove_reference_t<dsl::remove_var_wrapper_t<Arg>>;
        TArg targ = arg;

        if constexpr(std::is_same_v<TArg, i32> || std::is_same_v<TArg, u32> ||
                     std::is_same_v<TArg, i64> || std::is_same_v<TArg, u64> ||
                     std::is_same_v<TArg, f64>)
            return targ._load();
        else if constexpr(std::is_same_v<TArg, i16> || std::is_same_v<TArg, i8> ||
                          std::is_same_v<TArg, boolean> || std::is_same_v<TArg, char_t>)
            return i32(targ)._load();
        else if constexpr(std::is_same_v<TArg, u16> || std::is_same_v<TArg, u8>)
            return u32(targ)._load();
        else if constexpr(std::is_same_v<TArg, f32>)
            return f64(targ)._load();
        else
        {
            static_assert(dsl::is_cuj_pointer_v<TArg>);
            return targ._load();
        }
    }

} // namespace cstd_detail

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
    ((call.args.push_back(newRC<core::Expr>(
        cstd_detail::convert_print_arg(args)))), ...);

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
