/* Geometry Shader: Arrow Renderer
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;


void lighting( vec3 norm)
{
    // transformation of the normal into eye space
    normal = normalize(gl_NormalMatrix * norm);
    // normalize the direction of the light
    lightDir = normalize(vec3(gl_LightSource[0].position));
    // normalize the halfVector to pass it to the fragment shader
    halfVector = normalize(gl_LightSource[0].halfVector.xyz);
    // compute the diffuse, ambient and globalAmbient terms
    diffuse = gl_FrontColor * gl_LightSource[0].diffuse;
    ambient = gl_FrontColor * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_FrontColor;
}

void main(void)
{
    // get all vertex positions
    vec3 v0 = vec3(gl_PositionIn[0].xyz) / gl_PositionIn[0].w;
    vec3 v1 = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;
    vec3 v2 = vec3(gl_PositionIn[2].xyz) / gl_PositionIn[2].w;
    vec3 v3 = vec3(gl_PositionIn[3].xyz) / gl_PositionIn[3].w;
    vec3 v4 = vec3(gl_PositionIn[4].xyz) / gl_PositionIn[4].w;
    vec3 v5 = vec3(gl_PositionIn[5].xyz) / gl_PositionIn[5].w;
    // compute all needed directions
    vec3 dir20 = v2 - v0;
    vec3 dir12 = v1 - v2;
    vec3 dir32 = v3 - v2;
    vec3 dir43 = v4 - v3;
    vec3 dir53 = v5 - v3;

    // scale factor for the width of the tube
    float scale = gl_FrontColorIn[1].x;
    float stretch1 = gl_FrontColorIn[1].y;
    float stretch2 = gl_FrontColorIn[1].z;

    vec3 ortho1 = normalize( dir20 + dir32);
    vec3 ortho2 = normalize( dir32 + dir53);

    vec3 dir1 = normalize( cross( dir12, ortho1));
    vec3 dir2 = normalize( cross( dir43, ortho2));

    dir1 = normalize( cross( dir1, ortho1))*stretch1;
    dir2 = normalize( cross( dir2, ortho2))*stretch2;

    vec3 norm1 = normalize( cross( dir1, ortho1));
    vec3 norm2 = normalize( cross( dir2, ortho2));

    /////////////////////
    // draw top ribbon //
    /////////////////////
    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale).xyz, 1.0);
    lighting( normalize( norm1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale).xyz, 1.0);
    lighting( normalize( norm1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
    lighting( normalize( norm2));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
    lighting( normalize( norm2));
    EmitVertex();

    EndPrimitive();

    ////////////////////////
    // draw bottom ribbon //
    ////////////////////////
    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale).xyz, 1.0);
    lighting( normalize(-norm1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale).xyz, 1.0);
    lighting( normalize(-norm1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
    lighting( normalize(-norm2));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale).xyz, 1.0);
    lighting( normalize(-norm2));
    EmitVertex();

    EndPrimitive();

    /////////////////////
    // draw first side //
    /////////////////////
    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale).xyz, 1.0);
    lighting( normalize( dir1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale).xyz, 1.0);
    lighting( normalize( dir1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
    lighting( normalize( dir2));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale).xyz, 1.0);
    lighting( normalize( dir2));
    EmitVertex();

    EndPrimitive();

    //////////////////////
    // draw second side //
    //////////////////////
    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale).xyz, 1.0);
    lighting( normalize(-dir1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale).xyz, 1.0);
    lighting( normalize(-dir1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
    lighting( normalize(-dir2));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
    lighting( normalize(-dir2));
    EmitVertex();

    EndPrimitive();

    if( gl_FrontColorIn[2].y > 0.0 )
    {
        ////////////////
        // draw start //
        ////////////////
        gl_FrontColor = gl_FrontColorIn[3];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
        lighting( normalize( dir32));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[3];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
        lighting( normalize( dir32));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[3];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
        lighting( normalize( dir32));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[3];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale).xyz, 1.0);
        lighting( normalize( dir32));
        EmitVertex();

        EndPrimitive();
    }

    if( gl_FrontColorIn[2].z > 0.0 )
    {
        //////////////
        // draw end //
        //////////////
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale).xyz, 1.0);
        lighting( normalize(-dir20));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale).xyz, 1.0);
        lighting( normalize(-dir20));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale).xyz, 1.0);
        lighting( normalize(-dir20));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale).xyz, 1.0);
        lighting( normalize(-dir20));
        EmitVertex();

        EndPrimitive();
    }

}
