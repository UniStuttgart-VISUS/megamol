in vec3 colour;

out layout(location = 0) vec4 frag_colour;

void main(void)
{
    frag_colour = vec4(colour,1.0);
}
