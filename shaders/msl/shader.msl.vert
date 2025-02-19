#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct UBO
{
    float4x4 mvp[1];
};

struct main0_out
{
    float4 vColor [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 inPosition [[attribute(0)]];
    float4 inColor [[attribute(1)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant UBO& _20 [[buffer(0)]], uint gl_InstanceIndex [[instance_id]])
{
    main0_out out = {};
    out.gl_Position = _20.mvp[int(gl_InstanceIndex)] * float4(in.inPosition, 1.0);
    out.vColor = in.inColor;
    return out;
}

