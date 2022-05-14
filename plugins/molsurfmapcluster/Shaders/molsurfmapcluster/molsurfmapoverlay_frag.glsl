uniform sampler2D tex;
uniform vec3 viewvec;

in vec2 texCoord;


layout(location = 0) out vec4 out_color;

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() 
{
    float pi = 3.14159265358;
    vec3 defview = vec3(0.0, 0.0, -1.0);
    vec2 point = vec2(0.5, 0.5); // dummy 

    vec3 v = -normalize(viewvec);
    //float theta = acos(v.z); // we assume that r = 1
    //float phi = atan(v.x / v.z);

    //point.x = theta / pi;
    //point.y = sin(phi);

    vec2 ntexCoord = vec2(texCoord.x, 1.0 - texCoord.y);
    vec4 color = texture(tex, ntexCoord);

    float xangle = -(texCoord.x * 2.0 * pi + 0.5 * pi);
    float yval = texCoord.y * 2.0 - 1.0;
    vec3 pos;
    pos.x = cos(xangle);
    pos.z = sin(xangle);
    pos.y = yval;
    pos = normalize(pos);
    //v = vec3(1.0, 0.0, 0.0);
//    if(distance(ntexCoord, point) > 0.25) {
    if(dot(pos, v) < 0.05) {
        vec3 c = rgb2hsv(color.xyz);
        c.z = c.z * 0.5;
        c = hsv2rgb(c);
        color = vec4(c, 1.0); 
    }
    out_color = color;
}