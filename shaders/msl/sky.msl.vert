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

struct type_WorldVertexUniforms
{
    float4x4 mvp;
    float4x4 model_view;
};

struct type_SkyVertexUniforms
{
    float4x4 inv_view;
    float4x4 inv_proj;
};

struct type_FragmentUniforms
{
    float4 fog_color;
    float4 sky_top_color;
    float4 sky_horizon_color;
    float4 fog_distances;
};

constant spvUnsafeArray<float2, 3> _40 = spvUnsafeArray<float2, 3>({ float2(-1.0), float2(3.0, -1.0), float2(-1.0, 3.0) });

struct SkyVertexMain_out
{
    float3 out_var_TEXCOORD0 [[user(locn0)]];
    float4 gl_Position [[position]];
};

vertex SkyVertexMain_out SkyVertexMain(constant type_SkyVertexUniforms& SkyVertexUniforms [[buffer(1)]], uint gl_VertexIndex [[vertex_id]])
{
    SkyVertexMain_out out = {};
    out.gl_Position = float4(_40[gl_VertexIndex], 0.0, 1.0);
    out.out_var_TEXCOORD0 = (SkyVertexUniforms.inv_view * float4((SkyVertexUniforms.inv_proj * float4(_40[gl_VertexIndex], 1.0, 1.0)).xyz, 0.0)).xyz;
    return out;
}
