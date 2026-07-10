cbuffer UiUniforms : register(b0, space1)
{
    float4 ui_rect;
    float4 ui_color;
    float4 ui_corner_radii;
    float4 ui_border_widths;
    float2 ui_viewport;
    float2 ui_padding;
};

struct UiVertexInput
{
    uint vertex_id : SV_VertexID;
};

struct UiVertexOutput
{
    float4 position : SV_Position;
    float2 local    : TEXCOORD0;
};

UiVertexOutput UiVertexMain(UiVertexInput input)
{
    float2 corners[6] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 1.0),
    };

    UiVertexOutput output;
    float2 local01 = corners[input.vertex_id];
    float2 pixel = ui_rect.xy + local01 * ui_rect.zw;
    float2 ndc = pixel / ui_viewport * 2.0 - 1.0;
    ndc.y = -ndc.y;
    output.position = float4(ndc, 0.0, 1.0);
    output.local = local01 * ui_rect.zw;
    return output;
}

float UiCornerAlpha(float2 local, float2 size, float4 radii)
{
    float alpha = 1.0;
    if (radii.x > 0.0 && local.x < radii.x && local.y < radii.x) {
        alpha *= step(length(local - float2(radii.x, radii.x)), radii.x);
    }
    if (radii.y > 0.0 && local.x > size.x - radii.y && local.y < radii.y) {
        alpha *= step(length(local - float2(size.x - radii.y, radii.y)), radii.y);
    }
    if (radii.z > 0.0 && local.x < radii.z && local.y > size.y - radii.z) {
        alpha *= step(length(local - float2(radii.z, size.y - radii.z)), radii.z);
    }
    if (radii.w > 0.0 && local.x > size.x - radii.w && local.y > size.y - radii.w) {
        alpha *= step(length(local - float2(size.x - radii.w, size.y - radii.w)), radii.w);
    }
    return alpha;
}

float4 UiFragmentMain(UiVertexOutput input) : SV_Target0
{
    float2 size = ui_rect.zw;
    float4 radii = min(ui_corner_radii, min(size.x, size.y) * 0.5);
    float outer = UiCornerAlpha(input.local, size, radii);
    if (outer < 0.5) {
        discard;
    }

    float4 border = ui_border_widths;
    bool has_border = border.x > 0.0 || border.y > 0.0 || border.z > 0.0 || border.w > 0.0;
    if (has_border) {
        float2 inner_size = max(size - float2(border.x + border.y, border.z + border.w), 0.0);
        float2 inner_local = input.local - float2(border.x, border.z);
        float4 inner_radii = max(radii - float4(min(border.x, border.z), min(border.y, border.z), min(border.x, border.w), min(border.y, border.w)), 0.0);
        bool in_inner_rect = inner_local.x >= 0.0 && inner_local.y >= 0.0 && inner_local.x <= inner_size.x && inner_local.y <= inner_size.y;
        if (in_inner_rect && UiCornerAlpha(inner_local, inner_size, inner_radii) >= 0.5) {
            discard;
        }
    }

    return ui_color;
}
