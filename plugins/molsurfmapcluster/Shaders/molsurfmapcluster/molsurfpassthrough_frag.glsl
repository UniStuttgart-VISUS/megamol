layout(location = 0) out vec4 out_color;

uniform bool coloredmode = false;
uniform vec4 color;

in vec4 overridecolor;

void main() 
{
    out_color = color;
    if(coloredmode) {
        out_color = overridecolor;
    }
}