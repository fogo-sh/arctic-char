cbuffer VertexUniforms : register(b0, space1)
{
    float4x4 mvp;
};

struct VertexInput
{
    float3 position : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    float4 color    : TEXCOORD3;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color    : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
};

VertexOutput VertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(mvp, float4(input.position, 1.0));
    output.color = input.color;
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

float4 FragmentMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
