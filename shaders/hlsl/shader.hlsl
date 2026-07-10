cbuffer WorldVertexUniforms : register(b0, space1)
{
    float4x4 mvp;
    float4x4 model_view;
};

cbuffer SkyVertexUniforms : register(b1, space1)
{
    float4x4 inv_view;
    float4x4 inv_proj;
};

cbuffer FragmentUniforms : register(b0, space3)
{
    float4 fog_color;
    float4 sky_top_color;
    float4 sky_horizon_color;
    float4 fog_distances;
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
    float3 view_pos : TEXCOORD3;
};

VertexOutput VertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(mvp, float4(input.position, 1.0));
    output.color = input.color;
    output.normal = input.normal;
    output.uv = input.uv;
    output.view_pos = mul(model_view, float4(input.position, 1.0)).xyz;
    return output;
}

float4 FragmentMain(VertexOutput input) : SV_Target0
{
    float fog_amount = smoothstep(fog_distances.x, fog_distances.y, length(input.view_pos));
    return float4(lerp(input.color.rgb, fog_color.rgb, fog_amount), input.color.a);
}

struct SkyVertexInput
{
    uint vertex_id : SV_VertexID;
};

struct SkyVertexOutput
{
    float4 position : SV_Position;
    float3 ray_dir  : TEXCOORD0;
};

SkyVertexOutput SkyVertexMain(SkyVertexInput input)
{
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };

    SkyVertexOutput output;
    float2 ndc = positions[input.vertex_id];
    float4 view_ray = mul(inv_proj, float4(ndc, 1.0, 1.0));
    output.ray_dir = mul(inv_view, float4(view_ray.xyz, 0.0)).xyz;
    output.position = float4(ndc, 0.0, 1.0);
    return output;
}

float4 SkyFragmentMain(SkyVertexOutput input) : SV_Target0
{
    float3 ray = normalize(input.ray_dir);
    float height = saturate(ray.y * 0.5 + 0.5);
    float horizon = 1.0 - smoothstep(0.0, 0.35, abs(ray.y));
    float3 sky = lerp(sky_horizon_color.rgb, sky_top_color.rgb, height);
    sky = lerp(sky, fog_color.rgb, horizon * 0.35);
    return float4(sky, 1.0);
}
