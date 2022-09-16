layout(isolines, equal_spacing) in;

in vec4 myColor[];
out vec4 vertColor;

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;
    vec4 p3 = gl_in[3].gl_Position;

    vertColor = myColor[1];

    float u = gl_TessCoord.x;

    // Catmull-Rom Spline
    //gl_Position = 0.5 *( (2.0 * p1) + (-p0 + p2) * u + ( 2.0 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1- 3 * p2 + p3) * u*u*u);

    // Cubic B-Spline
    u += 3;
    float q = ( u - 1.0) / 3.0;
    vec4 d10 = p0 * ( 1.0 - q) + p1 * q;
    q = ( u - 2.0) / 3.0;
    vec4 d11 =  p1 * ( 1.0 - q) + p2 * q;
    q = ( u - 3.0) / 3.0;
    vec4 d12 =  p2 * ( 1.0 - q) + p3 * q;

    q = ( u - 2.0) / 2.0;
    vec4 d20 = d10 * ( 1.0 - q) + d11 * q;
    q = ( u - 3.0) / 2.0;
    vec4 d21 = d11 * ( 1.0 - q) + d12 * q;

    q = ( u - 3.0);
    gl_Position =  d20 * ( 1.0 - q) + d21 * q;
}
