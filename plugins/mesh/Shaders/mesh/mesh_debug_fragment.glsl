in vec3 normal;
in vec3 world_pos;

out layout(location = 0) vec4 frag_colour;

void main(void) {
    frag_colour = vec4(world_pos/vec3(50.0,50.0,-170.0),1.0);
}