uniform vec4 sphere;
uniform vec3 frontVertex;
uniform bool mirrorMap;

out int vertid;

#define PI  3.1415
       
void main(void) {
    gl_FrontColor = gl_Color;
    vertid = gl_VertexID;
    float len = length(gl_Vertex.xyz - sphere.xyz);
    if( abs(len - sphere.w) > 1.0 ) {
        gl_FrontColor = vec4(1.0, 1.0, 1.0, 1.0);
        len = 1.0 - len / sphere.w;
    } else {
        len = -0.1;
    }
    
    vec3 relCoord = normalize(gl_Vertex.xyz - sphere.xyz);
    vec3 relCoord2 = normalize(frontVertex - sphere.xyz);
    float lambda = sign(relCoord.x) * PI / 2.0;
    if( abs(relCoord.z) > 0.001 ) {
        lambda = atan(relCoord.x, relCoord.z);
    }
    float lambda2 = 0.0;
    if( abs(relCoord2.z) > 0.001 ) {
        lambda2 = atan(relCoord2.x, relCoord2.z);
    }
    gl_Position = vec4((lambda - lambda2) / PI, relCoord.y, len, 1.0);
    if(mirrorMap) {
        gl_Position.x = -gl_Position.x;
    }
}
