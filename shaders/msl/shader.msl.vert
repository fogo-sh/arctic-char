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

vertex main0_out main0(constant UBO& _47 [[buffer(0)]], uint gl_VertexIndex [[vertex_id]])
{
    main0_out out = {};
    float4 position;
    if (int(gl_VertexIndex) == 0)
    {
        position = float4(-0.5, -0.5, 0.0, 1.0);
    }
    else
    {
        if (int(gl_VertexIndex) == 1)
        {
            position = float4(0.0, 0.5, 0.0, 1.0);
        }
        else
        {
            if (int(gl_VertexIndex) == 2)
            {
                position = float4(0.5, -0.5, 0.0, 1.0);
            }
        }
    }
    out.gl_Position = _47.mvp * position;
    return out;
}

