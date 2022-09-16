//#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

uniform float streamTubeThicknessScl;
uniform float minColTexValue;
uniform float maxColTexValue;

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;

varying vec3 fragTangent;
varying vec3 fragView;

varying vec3 view;


void lightingStreamlines(vec3 norm) {
    // transformation of the normal into eye space
    //normal = normalize(gl_NormalMatrix * norm);
    normal = normalize(norm);
    // normalize the direction of the light
    lightDir = (gl_ModelViewMatrixInverse*normalize(gl_LightSource[0].position)).xyz;
    //lightDir = normalize(gl_LightSource[0].position + gl_ModelViewMatrix*vec4(view,1.0));
    //lightDir = -gl_LightSource[0].position;
    // normalize the halfVector to pass it to the fragment shader
    halfVector = normalize(gl_LightSource[0].halfVector.xyz);
    // compute the diffuse, ambient and globalAmbient terms
    diffuse = gl_FrontColor * gl_LightSource[0].diffuse;
    ambient = gl_FrontColor * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_FrontColor;
}

void main(void) {

    /*gl_FrontColor = gl_FrontColorIn[1];
    gl_Position = gl_ModelViewProjectionMatrix*gl_PositionIn[1];
    fragTangent = normalize(gl_PositionIn[2].xyz-gl_PositionIn[0].xyz);
    fragView = (gl_ModelViewMatrix*gl_PositionIn[1]).xyz;
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[2];
    gl_Position = gl_ModelViewProjectionMatrix*gl_PositionIn[2];
    fragTangent = normalize(gl_PositionIn[3].xyz-gl_PositionIn[1].xyz);
    fragView = (gl_ModelViewMatrix*gl_PositionIn[2]).xyz;
    EmitVertex();*/

    // get all vertex positions
    //vec3 v0 = vec3(gl_PositionIn[0].xyz) / gl_PositionIn[0].w;
    vec3 v0 = vec3(gl_PositionIn[0].xyz);
    //vec3 v1 = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;
    //vec3 v2 = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;
    vec3 v2 = vec3(gl_PositionIn[1].xyz);
    //vec3 v3 = vec3(gl_PositionIn[2].xyz) / gl_PositionIn[2].w;
    vec3 v3 = vec3(gl_PositionIn[2].xyz);
    //vec3 v4 = (-gl_ModelViewMatrix*gl_PositionIn[3]).xyz;
    //vec3 v4 = vec3(gl_PositionIn[4].xyz) / gl_PositionIn[4].w;
    //vec3 v5 = vec3(gl_PositionIn[3].xyz) / gl_PositionIn[3].w;
    vec3 v5 = vec3(gl_PositionIn[3].xyz);

    vec3 camPos = gl_ModelViewMatrixInverse[3]; // (C) by Christoph

    // compute all needed directions
    vec3 dir20 = normalize(v2 - v0);
    //vec3 dir12 = v1 - v2;
    //vec3 dir12 = (-gl_ModelViewMatrix*gl_PositionIn[1]).xyz;
    //vec3 dir12 = normalize((-gl_ModelViewMatrix*vec4(v2, 1.0)).xyz);
    vec3 dir12 = normalize(camPos - v2);
    vec3 dir32 = normalize(v3 - v2);
    //vec3 dir43 = v4 - v3;
    //vec3 dir43 = (-gl_ModelViewMatrix*gl_PositionIn[2]).xyz;
    //vec3 dir43 = normalize((-gl_ModelViewMatrix*vec4(v3, 1.0)).xyz);
    vec3 dir43 = normalize(camPos - v3);
    vec3 dir53 = normalize(v5 - v3);

    // number of segments
    const int n = 15;

    // scale factor for the width of the tube
    //float scale = length(gl_FrontColorIn[0].rgb)*gl_FrontColorIn[1].a*streamTubeThicknessScl;
    //float scale = 0.3;
    //float stretch1 = gl_FrontColorIn[1].y;
    //float stretch1 = length(gl_FrontColorIn[1].rgb)*gl_FrontColorIn[1].a*streamTubeThicknessScl;
    //float stretch2 = gl_FrontColorIn[1].z;
    //float stretch2 = length(gl_FrontColorIn[2].rgb)*gl_FrontColorIn[2].a*streamTubeThicknessScl;
    float stretch1 = streamTubeThicknessScl/100.;
    float stretch2 = streamTubeThicknessScl/100.;


    vec3 ortho1 = normalize(dir20 + dir32);
    vec3 ortho2 = normalize(dir32 + dir53);

    vec3 dir1 = normalize(cross(dir12, ortho1));
    vec3 dir2 = normalize(cross(dir43, ortho2));

    // angle for the rotation
    float alpha;
    // matrices for rotation
    mat3 e = mat3( 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    mat3 m1 = mat3( 0.0,-ortho1.z, ortho1.y, ortho1.z, 0.0,-ortho1.x,-ortho1.y, ortho1.x, 0.0);
    mat3 m2 = mat3( 0.0,-ortho2.z, ortho2.y, ortho2.z, 0.0,-ortho2.x,-ortho2.y, ortho2.x, 0.0);
    mat3 m1quat = m1 * m1;
    mat3 m2quat = m2 * m2;

    vec3 res1;
    vec3 res2;

    vec4 colTest1 = vec4(vec3((dir12.y+1.0)/2.0), 1.0);
    vec4 colTest2 = vec4(vec3((dir43.y+1.0)/2.0), 1.0);

    dir1 = dir1 * stretch1;
    dir2 = dir2 * stretch2;

    // Compute colors based on velocity
    float val1 = length(gl_FrontColorIn[1].rgb);
    vec3 colMsh1 = CoolWarmMsh(val1, minColTexValue, (minColTexValue+maxColTexValue)*0.5, maxColTexValue);
    vec4 col1 = vec4(MSH2RGB(colMsh1.x, colMsh1.y, colMsh1.z), 1.0);
    float val2 = length(gl_FrontColorIn[2].rgb);
    vec3 colMsh2 = CoolWarmMsh(val2, minColTexValue, (minColTexValue+maxColTexValue)*0.5, maxColTexValue);
    vec4 col2 = vec4(MSH2RGB(colMsh2.x, colMsh2.y, colMsh2.z), 1.0);

    for( int i = 0; i < n; i++ ) {
        alpha = (360.0/float(n))*float(i);
        res1 = ( e + sin( radians( alpha)) * m1 + ( 1.0 - cos( radians( alpha))) * m1quat ) * dir1;
        res2 = ( e + sin( radians( alpha)) * m2 + ( 1.0 - cos( radians( alpha))) * m2quat ) * dir2;

        // copy color
        gl_FrontColor = col1;
        // copy position
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + res1).xyz, 1.0);
        // compute lighting
        view = dir12;
        lightingStreamlines( normalize( res1));

        // done with the vertex
        EmitVertex();

        gl_FrontColor = col2;
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + res2).xyz, 1.0);
        view = dir43;
        lightingStreamlines( normalize( res2));

        EmitVertex();
    }
    gl_FrontColor = col1;
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1).xyz, 1.0);
    view = dir12;
    lightingStreamlines( normalize( dir1));

    EmitVertex();

    gl_FrontColor = col2;
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2).xyz, 1.0);
    view = dir43;
    lightingStreamlines( normalize( dir2));

    EmitVertex();

    EndPrimitive();
}
