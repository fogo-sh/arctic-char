#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];

    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }

    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }

    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }

    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct type_UiUniforms
{
    float4 ui_rect;
    float4 ui_color;
    float4 ui_corner_radii;
    float4 ui_border_widths;
    float2 ui_viewport;
    float2 ui_padding;
};

constant spvUnsafeArray<float2, 6> _33 = spvUnsafeArray<float2, 6>({ float2(0.0), float2(1.0, 0.0), float2(0.0, 1.0), float2(1.0, 0.0), float2(1.0), float2(0.0, 1.0) });

struct UiVertexMain_out
{
    float2 out_var_TEXCOORD0 [[user(locn0)]];
    float4 gl_Position [[position]];
};

vertex UiVertexMain_out UiVertexMain(constant type_UiUniforms& UiUniforms [[buffer(0)]], uint gl_VertexIndex [[vertex_id]])
{
    UiVertexMain_out out = {};
    float2 _43 = _33[gl_VertexIndex] * UiUniforms.ui_rect.zw;
    float2 _49 = (((UiUniforms.ui_rect.xy + _43) / UiUniforms.ui_viewport) * 2.0) - float2(1.0);
    out.gl_Position = float4(_49.x, -_49.y, 0.0, 1.0);
    out.out_var_TEXCOORD0 = _43;
    return out;
}
