#version 430
///////////////
// GEOMETRY //
//////////////

////////////
// LAYOUT //
////////////
layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;

/////////
// IN //
//////// 
in vec4 color[];
in vec3 n[];
in vec3 uncVal[];              // Uncertainty and DITHERING values - from tess eval
    
//////////
// OUT //
/////////  
out vec4 mycol;
out vec3 normal;
out vec3 basenormal;
out vec3 uncValues;          // Uncertainty and DITHERING values  - pass to frag shader

/////////////////////
// INPUT variables //
/////////////////////
uniform mat4 MVP;  
uniform mat4 MVinvtrans;

//////////
// MAIN //
//////////
void main() {
    for(int i = 0; i < gl_in.length(); i++) {                       // length = ?
        gl_Position  = MVP * gl_in[i].gl_Position;
        mycol        = color[i];
        vec4 normal4 = MVinvtrans * vec4(n[i], 0);
        normal       = normalize(normal4.xyz);
        basenormal   = normalize(n[i]);
        uncValues    = uncVal[i];
       
        EmitVertex();
    }
    EndPrimitive();
}
