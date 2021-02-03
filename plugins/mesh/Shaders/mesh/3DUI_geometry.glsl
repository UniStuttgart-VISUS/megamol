layout (triangles) in;
layout (triangle_strip, max_vertices = 15) out; // 3 original + 4 per line

in vec3 vWorldPos[3];
in vec3 vNormal[3];
in vec3 vColor[3];
in flat int vID[3];

out vec3 worldPos;
out vec3 normal;
out vec3 colour;
out flat int id;

void main() 
{    
    gl_Position = gl_in[0].gl_Position;
    //colour = vColour[0];
    worldPos = vWorldPos[0];
    normal = vNormal[0];
    colour = vColor[0];
    id = vID[0];
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    //colour = vColour[1];
    worldPos = vWorldPos[1];
    normal = vNormal[1];
    colour = vColor[1];
    id = vID[1];
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    //colour = vColour[2];
    worldPos = vWorldPos[2];
    normal = vNormal[2];
    colour = vColor[2];
    id = vID[2];
    EmitVertex();
    EndPrimitive();

/*
    vec4 e10 =  gl_in[0].gl_Position - gl_in[1].gl_Position;
    vec4 e20 =  gl_in[0].gl_Position - gl_in[2].gl_Position;
    vec4 offset0 = vec4(normalize(cross(vec3(e10),vec3(e20))),0.0f);

    vec4 e01 =  gl_in[1].gl_Position - gl_in[0].gl_Position;
    vec4 e21 =  gl_in[1].gl_Position - gl_in[2].gl_Position;
    vec4 offset1 = vec4(normalize(cross(vec3(e21),vec3(e01))),0.0f);

    vec4 e02 =  gl_in[2].gl_Position - gl_in[0].gl_Position;
    vec4 e12 =  gl_in[2].gl_Position - gl_in[1].gl_Position;
    vec4 offset2 = vec4(normalize(cross(vec3(e02),vec3(e12))),0.0f);

    offset0 *= -0.00005f;
    offset1 *= -0.00005f;
    offset2 *= -0.00005f;

    float line_width = 0.01f;

    gl_Position = gl_in[0].gl_Position + offset2;
    worldPos = vWorldPos[0] + offset2.xyz;
    normal = vNormal[0];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + e02 * line_width + offset2;
    worldPos = vWorldPos[0]  + e02.xyz * line_width + offset2.xyz;
    normal = vNormal[0];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + offset2;
    worldPos = vWorldPos[1]  + offset2.xyz;
    normal = vNormal[1];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + e12 * line_width + offset2;
    worldPos = vWorldPos[1]  + e12.xyz * line_width + offset2.xyz;
    normal = vNormal[1];
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();

    gl_Position = gl_in[1].gl_Position + offset0;
    worldPos = vWorldPos[1] + offset0.xyz;
    normal = vNormal[1];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + e10 * line_width + offset0;
    worldPos = vWorldPos[1] + e10.xyz * line_width + offset0.xyz;
    normal = vNormal[1];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + offset0;
    worldPos = vWorldPos[2] + offset0.xyz;
    normal = vNormal[2];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + e20 * line_width + offset0;
    worldPos = vWorldPos[2] + e20.xyz * line_width + offset0.xyz;
    normal = vNormal[2];
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();

    gl_Position = gl_in[2].gl_Position + offset1;
    worldPos = vWorldPos[2] + offset1.xyz;
    normal = vNormal[2];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + e21 * line_width + offset1;
    worldPos = vWorldPos[2] + e21.xyz * line_width + offset1.xyz;
    normal = vNormal[2];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + offset1;
    worldPos = vWorldPos[0] + offset1.xyz;
    normal = vNormal[0];
    colour = vec3(1.0f);
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + e01 * line_width + offset1;
    worldPos = vWorldPos[0] + e01.xyz * line_width + offset1.xyz;
    normal = vNormal[0];
    colour = vec3(1.0f);
    EmitVertex();
    EndPrimitive();
*/
}