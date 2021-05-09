#include <cuj/builtin/math/atomic.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::atomic)

namespace detail
{

    template<typename T>
    class InternalAtomicAdd : public ast::InternalArithmeticValue<T>
    {
    public:

        RC<ast::InternalPointerValue<T>>    dst;
        RC<ast::InternalArithmeticValue<T>> val;

        ir::BasicValue gen_ir(ir::IRBuilder &builder) const override
        {
            auto dst_val = dst->gen_ir(builder);
            auto val_val = val->gen_ir(builder);

            auto ret_type = get_current_context()->get_type<T>();
            auto ret = builder.gen_temp_value(ret_type);

            const std::string name = std::string("atomic.add.")
                                   + (std::is_same_v<T, float> ? "f32" : "f64");
            auto op = ir::IntrinsicOp{ std::move(name), { dst_val, val_val } };

            builder.append_assign(ret, op);
            return ret;
        }
    };

} // namespace detail

f32 atomic_add(const Pointer<f32> &dst, const f32 &val)
{
    auto impl = newRC<detail::InternalAtomicAdd<float>>();
    impl->dst = dst.get_impl();
    impl->val = val.get_impl();

    f32 ret(std::move(impl));
    return ret;
}

f64 atomic_add(const Pointer<f64> &dst, const f64 &val)
{
    auto impl = newRC<detail::InternalAtomicAdd<double>>();
    impl->dst = dst.get_impl();
    impl->val = val.get_impl();

    f64 ret(std::move(impl));
    return ret;
}

CUJ_NAMESPACE_END(cuj::builtin::atomic)
