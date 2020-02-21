uniform sampler2D tex;

in vec2 texCoord;

layout(location = 0) out vec4 out_color;


void main() 
{
    vec4 in_color = texture(tex, texCoord);
    out_color = in_color;
    //out_color = vec4(1,0,0,1);
    //out_color = vec4(texCoord, 0, 1);
}