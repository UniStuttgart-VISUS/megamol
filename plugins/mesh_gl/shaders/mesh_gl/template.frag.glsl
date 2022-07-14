#version 430

// Data passed from previous shader stages
in vec3 world_normal;

// Forward rendering uses a single RGBA color output
out layout(location = 0) vec4 frag_colour;

void main(void) {
   // Typically, the fragment shader for a forward renderer will be much more complicated
   // but the minimum is to write some color to the output.
   frag_colour = vec4(1.0,1.0,1.0,1.0);
}
