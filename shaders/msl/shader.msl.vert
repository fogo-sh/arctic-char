#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct UBO
{
    float4x4 mvp;
};

struct main0_out
{
    float4 gl_Position [[position]];
};

struct main0_in
{
    float4 inPosition [[attribute(0)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant UBO& _19 [[buffer(0)]])
{
    main0_out out = {};
    out.gl_Position = _19.mvp * in.inPosition;
    return out;
}

