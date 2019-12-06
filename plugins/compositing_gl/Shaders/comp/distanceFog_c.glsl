uniform float density;
uniform float zNear;
uniform float zFar;

uniform sampler2D depth_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// https://docs.microsoft.com/en-us/windows/win32/direct3d9/fog-formulas
void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize (tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    float depth    = texelFetch(depth_tx2D, pixel_coords, 0).r;
    float f = 1.0;

    if(depth > 0.0)
    {
        float z_ndc = 2.0 * depth - 1.0;
        float linearDepth = 2.0 * zNear * zFar / (zFar + zNear - z_ndc * (zFar - zNear));
        linearDepth /= (zFar - zNear);
        f = 1.0 / exp2(linearDepth * density);

        //f = (1.0 - linearDepth);
    }
    imageStore(tgt_tx2D, pixel_coords, vec4(f, f, f, 1.0));
}