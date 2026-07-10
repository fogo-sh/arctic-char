#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_SlugVertexUniforms
{
    float4x4 mvp;
    float2 viewport;
};

struct type_SlugFragmentUniforms
{
    float weightBoost;
};

constant float _50 = {};

struct TextVertexMain_out
{
    float4 out_var_TEXCOORD0 [[user(locn0)]];
    float2 out_var_TEXCOORD1 [[user(locn1)]];
    float4 out_var_TEXCOORD2 [[user(locn2)]];
    int4 out_var_TEXCOORD3 [[user(locn3)]];
    float4 gl_Position [[position]];
};

struct TextVertexMain_in
{
    float4 in_var_TEXCOORD0 [[attribute(0)]];
    float4 in_var_TEXCOORD1 [[attribute(1)]];
    float4 in_var_TEXCOORD2 [[attribute(2)]];
    float4 in_var_TEXCOORD3 [[attribute(3)]];
    float4 in_var_TEXCOORD4 [[attribute(4)]];
};

vertex TextVertexMain_out TextVertexMain(TextVertexMain_in in [[stage_in]], constant type_SlugVertexUniforms& SlugVertexUniforms [[buffer(0)]])
{
    TextVertexMain_out out = {};
    float2 _87 = fast::normalize(in.in_var_TEXCOORD0.zw);
    float2 _88 = float4(SlugVertexUniforms.mvp[3u][0u], SlugVertexUniforms.mvp[3u][1u], _50, SlugVertexUniforms.mvp[3u][3u]).xy;
    float _91 = dot(_88, in.in_var_TEXCOORD0.xy) + SlugVertexUniforms.mvp[3u][3u];
    float _92 = dot(_88, _87);
    float2 _93 = float4(SlugVertexUniforms.mvp[0u][0u], SlugVertexUniforms.mvp[0u][1u], _50, SlugVertexUniforms.mvp[0u][3u]).xy;
    float _101 = ((_91 * dot(_93, _87)) - (_92 * (dot(_93, in.in_var_TEXCOORD0.xy) + SlugVertexUniforms.mvp[0u][3u]))) * SlugVertexUniforms.viewport.x;
    float2 _102 = float4(SlugVertexUniforms.mvp[1u][0u], SlugVertexUniforms.mvp[1u][1u], _50, SlugVertexUniforms.mvp[1u][3u]).xy;
    float _110 = ((_91 * dot(_102, _87)) - (_92 * (dot(_102, in.in_var_TEXCOORD0.xy) + SlugVertexUniforms.mvp[1u][3u]))) * SlugVertexUniforms.viewport.y;
    float _112 = _91 * _92;
    float _115 = (_101 * _101) + (_110 * _110);
    float2 _122 = in.in_var_TEXCOORD0.zw * (((_91 * _91) * (_112 + sqrt(_115))) / (_115 - (_112 * _112)));
    float2 _123 = in.in_var_TEXCOORD0.xy + _122;
    float _135 = _123.x;
    float _137 = _123.y;
    uint2 _155 = as_type<uint2>(in.in_var_TEXCOORD1.zw);
    uint _156 = _155.x;
    uint _161 = _155.y;
    out.gl_Position = float4(((_135 * SlugVertexUniforms.mvp[0u][0u]) + (_137 * SlugVertexUniforms.mvp[0u][1u])) + SlugVertexUniforms.mvp[0u][3u], ((_135 * SlugVertexUniforms.mvp[1u][0u]) + (_137 * SlugVertexUniforms.mvp[1u][1u])) + SlugVertexUniforms.mvp[1u][3u], ((_135 * SlugVertexUniforms.mvp[2u][0u]) + (_137 * SlugVertexUniforms.mvp[2u][1u])) + SlugVertexUniforms.mvp[2u][3u], ((_135 * SlugVertexUniforms.mvp[3u][0u]) + (_137 * SlugVertexUniforms.mvp[3u][1u])) + SlugVertexUniforms.mvp[3u][3u]);
    out.out_var_TEXCOORD0 = in.in_var_TEXCOORD4;
    out.out_var_TEXCOORD1 = float2(in.in_var_TEXCOORD1.x + dot(_122, in.in_var_TEXCOORD2.xy), in.in_var_TEXCOORD1.y + dot(_122, in.in_var_TEXCOORD2.zw));
    out.out_var_TEXCOORD2 = in.in_var_TEXCOORD3;
    out.out_var_TEXCOORD3 = int4(int(_156 & 65535u), int(_156 >> 16u), int(_161 & 65535u), int(_161 >> 16u));
    return out;
}
