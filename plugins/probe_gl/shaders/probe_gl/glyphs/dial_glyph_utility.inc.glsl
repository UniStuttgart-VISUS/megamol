bool highlightCorners(vec2 uv_coords){
    return ( (uv_coords.x > 0.99 && uv_coords.x > uv_coords.y && uv_coords.y > 0.9) ||
        (uv_coords.y > 0.99 && uv_coords.x < uv_coords.y && uv_coords.x > 0.9) ||
        (uv_coords.x < 0.01 && uv_coords.x < uv_coords.y && uv_coords.y < 0.05) ||
        (uv_coords.y < 0.01 && uv_coords.x > uv_coords.y && uv_coords.x < 0.05) );
};

//http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

void tiltGlyph(inout vec3 glyph_right, inout vec3 glyph_up, in vec3 probe_direction, in vec3 camera_front){
    float angle = acos(abs(dot(probe_direction, camera_front)));
    vec3 axis = normalize(cross(probe_direction, camera_front));

    mat4 rot = rotationMatrix(axis,angle);

    vec3 tilted_glyph_up = (rot * vec4(glyph_up,1.0)).xyz;
    vec3 tilted_glyph_right = (rot * vec4(glyph_right,1.0)).xyz;

    float tilt_factor = (angle / 1.57);
    tilt_factor = pow(tilt_factor,4.0);
    tilt_factor = 1.0 - tilt_factor;
    glyph_up = normalize(mix(tilted_glyph_up, glyph_up, tilt_factor));
    glyph_right = normalize(mix(tilted_glyph_right, glyph_right, tilt_factor));
};

vec3 fakeViridis(float lerp)
{
    vec3 c0 = vec3(0.2823645529290169,0.0,0.3310101940118055);
    vec3 c1 = vec3(0.24090172204161298,0.7633448774061599,0.42216355577803744);
    vec3 c2 = vec3(0.9529994532916154,0.9125452328290099,0.11085876909361342);

    return lerp < 0.5 ? mix(c0,c1,lerp * 2.0) : mix(c1,c2,(lerp*2.0)-1.0);
};

// Compute clockwise angle of polar coordinates starting at south normalized to [0,1]
float computeNormalizedAngle(vec2 uv_coords){
    vec2 pixel_direction = normalize(uv_coords * 2.0 - vec2(1.0,1.0));
    float angle = atan(
        pixel_direction.x,
        pixel_direction.x > 0.0 ? -pixel_direction.y : pixel_direction.y
    );
    angle = pixel_direction.x < 0.0 ? angle * -1.0 + 3.14159 : angle;
    return (1.0 - (angle / 6.283185 /*2pi*/));
};
