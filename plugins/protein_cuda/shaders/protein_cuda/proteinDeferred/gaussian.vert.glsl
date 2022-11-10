#version 110

uniform vec2 screenResInv;

void main(void)
{
    // this is half a pixel with and height because
    // the screen-aligned quad has a width and height of 2
    vec2 halfPixelSize = 0.5*screenResInv;

    // Clean up inaccuracies
    vec2 pos = sign(gl_Vertex.xy);
    gl_Position = vec4(pos, 0.0, 1.0);

    // offset to properly align pixels with texels
    //gl_Position.xy += vec2(-1.0, -1.0) * halfPixelSize;

    gl_TexCoord[0].xy = 0.5 * pos + 0.5;
    //gl_TexCoord[0].y = 1.0 - gl_TexCoord[0].y;
}
