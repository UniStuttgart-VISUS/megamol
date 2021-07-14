layout (triangles) in;
layout (triangle_strip, max_vertices = 15) out; // 3 original + 4 per line

in vec3 vColour[3];

out vec3 colour;

void main() 
{    
    gl_Position = gl_in[0].gl_Position;
    colour = vColour[0];
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    colour = vColour[1];
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    colour = vColour[2];
    EmitVertex();
    EndPrimitive();


    vec4 e10 =  gl_in[0].gl_Position - gl_in[1].gl_Position;
    vec4 e20 =  gl_in[0].gl_Position - gl_in[2].gl_Position;
    vec4 offset0 = vec4(normalize(cross(vec3(e10),vec3(e20))),0.0f);

    vec4 e01 =  gl_in[1].gl_Position - gl_in[0].gl_Position;
    vec4 e21 =  gl_in[1].gl_Position - gl_in[2].gl_Position;
    vec4 offset1 = vec4(normalize(cross(vec3(e21),vec3(e01))),0.0f);

    vec4 e02 =  gl_in[2].gl_Position - gl_in[0].gl_Position;
    vec4 e12 =  gl_in[2].gl_Position - gl_in[1].gl_Position;
    vec4 offset2 = vec4(normalize(cross(vec3(e02),vec3(e12))),0.0f);

    offset0 *= 0.00005f;
    offset1 *= 0.00005f;
    offset2 *= 0.00005f;

    float line_width = 0.01f;

    gl_Position = gl_in[0].gl_Position + offset2;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + e02 * line_width + offset2;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + offset2;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + e12 * line_width + offset2;
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();

    gl_Position = gl_in[1].gl_Position + offset0;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + e10 * line_width + offset0;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + offset0;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + e20 * line_width + offset0;
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();

    gl_Position = gl_in[2].gl_Position + offset1;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + e21 * line_width + offset1;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + offset1;
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + e01 * line_width + offset1;
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();
}