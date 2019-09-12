uniform sampler2D src_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// see http://horde3d.org/wiki/index.php5?title=Shading_Technique_-_FXAA
void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize (tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_resolution);

    float FXAA_SPAN_MAX = 8.0;
    float FXAA_REDUCE_MUL = 1.0/8.0;
    float FXAA_REDUCE_MIN = 1.0/128.0;
    vec3 rgbNW = texelFetch(src_tx2D, pixel_coords + ivec2(-1,-1),0).xyz;
    vec3 rgbNE = texelFetch(src_tx2D, pixel_coords + ivec2(1,-1) ,0).xyz;
    vec3 rgbSW = texelFetch(src_tx2D, pixel_coords + ivec2(-1,1) ,0).xyz;
    vec3 rgbSE = texelFetch(src_tx2D, pixel_coords + ivec2(1,1)  ,0).xyz;
    vec3 rgbM  = texelFetch(src_tx2D, pixel_coords, 0).xyz;
    
    vec3 luma=vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
    float dirReduce = max(
            (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
            FXAA_REDUCE_MIN);
      
    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);
    
    dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) / vec2(tgt_resolution);
            
    vec3 rgbA = (1.0/2.0) * (
            texture(src_tx2D, pixel_coords_norm.xy + dir * (1.0/3.0 - 0.5)).xyz +
            texture(src_tx2D, pixel_coords_norm.xy + dir * (2.0/3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
            texture(src_tx2D, pixel_coords_norm.xy + dir * (0.0/3.0 - 0.5)).xyz +
            texture(src_tx2D, pixel_coords_norm.xy + dir * (3.0/3.0 - 0.5)).xyz);
    float lumaB = dot(rgbB, luma);

    if((lumaB < lumaMin) || (lumaB > lumaMax)){
        imageStore(tgt_tx2D, pixel_coords , vec4(rgbA,1.0) );
    }
    else{
        imageStore(tgt_tx2D, pixel_coords , vec4(rgbB,1.0) );
    }
}