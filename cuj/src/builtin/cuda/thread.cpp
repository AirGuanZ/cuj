#if CUJ_ENABLE_CUDA

#include <cuj/builtin/cuda/thread.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::cuda)

namespace
{

    const char *get_intrinsic_name(IntrinsicValueType type)
    {
#define INTRINSIC_CASE(NAME) case IntrinsicValueType::NAME: return "cuda." #NAME

        switch(type)
        {
        INTRINSIC_CASE(thread_index_x);
        INTRINSIC_CASE(thread_index_y);
        INTRINSIC_CASE(thread_index_z);
        INTRINSIC_CASE(block_index_x);
        INTRINSIC_CASE(block_index_y);
        INTRINSIC_CASE(block_index_z);
        INTRINSIC_CASE(block_dim_x);
        INTRINSIC_CASE(block_dim_y);
        INTRINSIC_CASE(block_dim_z);
        }

        return "unknown";

#undef INTRINSIC_CASE
    }

    Value<int> intrinsic_int_value(IntrinsicValueType type)
    {
        auto impl = newRC<detail::InternalIntrinsicIntValue>();
        impl->type = type;
        return Value<int>(std::move(impl));
    }

    class ThreadBlockBarrierStatement : public ast::Statement
    {
    public:

        void gen_ir(ir::IRBuilder &builder) const override
        {
            auto call = ir::IntrinsicCall{ { "cuda.thread_block_barrier" } };
            builder.append_statement(newRC<ir::Statement>(call));
        }
    };

} // namespace anonymous

ir::BasicValue detail::InternalIntrinsicIntValue::gen_ir(
    ir::IRBuilder &builder) const
{
    ir::IntrinsicOp op = { get_intrinsic_name(type), {} };

    auto int_type = get_current_context()->get_type<int>();
    auto ret = builder.gen_temp_value(int_type);

    builder.append_assign(ret, op);
    return ret;
}

Value<int> thread_index_x()
{
    return intrinsic_int_value(IntrinsicValueType::thread_index_x);
}

Value<int> thread_index_y()
{
    return intrinsic_int_value(IntrinsicValueType::thread_index_y);
}

Value<int> thread_index_z()
{
    return intrinsic_int_value(IntrinsicValueType::thread_index_z);
}

Value<int> block_index_x()
{
    return intrinsic_int_value(IntrinsicValueType::block_index_x);
}

Value<int> block_index_y()
{
    return intrinsic_int_value(IntrinsicValueType::block_index_y);
}

Value<int> block_index_z()
{
    return intrinsic_int_value(IntrinsicValueType::block_index_z);
}

Value<int> block_dim_x()
{
    return intrinsic_int_value(IntrinsicValueType::block_dim_x);
}

Value<int> block_dim_y()
{
    return intrinsic_int_value(IntrinsicValueType::block_dim_y);
}

Value<int> block_dim_z()
{
    return intrinsic_int_value(IntrinsicValueType::block_dim_z);
}

Dim3 thread_index()
{
    return Dim3(thread_index_x(), thread_index_y(), thread_index_z());
}

Dim3 block_index()
{
    return Dim3(block_index_x(), block_index_y(), block_index_z());
}

Dim3 block_dim()
{
    return Dim3(block_dim_x(), block_dim_y(), block_dim_z());
}

void sync_block_threads()
{
    auto stat = newRC<ThreadBlockBarrierStatement>();
    get_current_function()->append_statement(std::move(stat));
}

CUJ_NAMESPACE_END(cuj::builtin::cuda)

#endif // #if CUJ_ENABLE_CUDA
