#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct TextUBO
{
    float4x4 ortho;
};

struct main0_out
{
    float4 gl_Position [[position]];
};

struct main0_in
{
    float4 inPosition [[attribute(0)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant TextUBO& ubo [[buffer(0)]])
{
    main0_out out = {};
    out.gl_Position = ubo.ortho * in.inPosition;
    return out;
}

