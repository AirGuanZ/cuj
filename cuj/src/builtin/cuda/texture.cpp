#if CUJ_ENABLE_CUDA

#include <cuj/builtin/cuda/texture.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::cuda)

namespace
{

    template<typename OutputComponent>
    class InternalIntrinsicSampleTexture : public ast::Statement
    {
    public:

        std::string                                                 name;
        RC<TextureObject::ImplType>                                 tex;
        std::vector<RC<ast::InternalArithmeticValue<float>>>        tex_coords;
        std::vector<RC<ast::InternalPointerValue<OutputComponent>>> outputs;

        void gen_ir(ir::IRBuilder &builder) const override
        {
            auto call = ir::IntrinsicCall{ { name, {} } };
            call.op.args.push_back(tex->gen_ir(builder));
            for(auto &c : tex_coords)
                call.op.args.push_back(c->gen_ir(builder));
            for(auto &o : outputs)
                call.op.args.push_back(o->gen_ir(builder));

            builder.append_statement(newRC<ir::Statement>(call));
        }
    };

} // namespace anonymous

math::Vec1f sample_texture2d_1f(TextureObject tex, f32 u, f32 v)
{
    math::Vec4f t = sample_texture2d_4f(tex, u, v);
    return math::make_vec1f(t.x);
}

math::Vec2f sample_texture2d_2f(TextureObject tex, f32 u, f32 v)
{
    math::Vec4f t = sample_texture2d_4f(tex, u, v);
    return math::make_vec2f(t.x, t.y);
}

math::Vec3f sample_texture2d_3f(TextureObject tex, f32 u, f32 v)
{
    math::Vec4f t = sample_texture2d_4f(tex, u, v);
    return math::make_vec3f(t.x, t.y, t.z);
}

math::Vec4f sample_texture2d_4f(TextureObject tex, f32 u, f32 v)
{
    f32 r, g, b, a;
    
    auto stat = newRC<InternalIntrinsicSampleTexture<float>>();
    stat->name       = "cuda.sample.2d.f32";
    stat->tex        = tex.get_impl();
    stat->tex_coords = { u.get_impl(), v.get_impl() };
    stat->outputs    = {
        r.address().get_impl(), g.address().get_impl(),
        b.address().get_impl(), a.address().get_impl() };
    get_current_function()->append_statement(std::move(stat));

    return math::make_vec4f(r, g, b, a);
}

math::Vec1i sample_texture2d_1i(TextureObject tex, f32 u, f32 v)
{
    math::Vec4i t = sample_texture2d_4i(tex, u, v);
    return math::make_vec1i(t.x);
}

math::Vec2i sample_texture2d_2i(TextureObject tex, f32 u, f32 v)
{
    math::Vec4i t = sample_texture2d_4i(tex, u, v);
    return math::make_vec2i(t.x, t.y);
}

math::Vec3i sample_texture2d_3i(TextureObject tex, f32 u, f32 v)
{
    math::Vec4i t = sample_texture2d_4i(tex, u, v);
    return math::make_vec3i(t.x, t.y, t.z);
}

math::Vec4i sample_texture2d_4i(TextureObject tex, f32 u, f32 v)
{
    i32 r, g, b, a;
    
    auto stat = newRC<InternalIntrinsicSampleTexture<int>>();
    stat->name       = "cuda.sample.2d.i32";
    stat->tex        = tex.get_impl();
    stat->tex_coords = { u.get_impl(), v.get_impl() };
    stat->outputs    = {
        r.address().get_impl(), g.address().get_impl(),
        b.address().get_impl(), a.address().get_impl() };
    get_current_function()->append_statement(std::move(stat));

    return math::make_vec4i(r, g, b, a);
}

CUJ_NAMESPACE_END(cuj::builtin::cuda)

#endif // #if CUJ_ENABLE_CUDA
