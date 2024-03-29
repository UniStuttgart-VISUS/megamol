/* Geometry Shader: Cartoon Renderer
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable


void lighting( vec3 norm)
{
    vec3 normal, lightDir;
    vec4 diffuse, ambient, globalAmbient;
    float NdotL;

    normal = normalize(gl_NormalMatrix * norm);
    lightDir = normalize(vec3(gl_LightSource[0].position));
    NdotL = max(dot(normal, lightDir), 0.0);
    diffuse = gl_FrontColor * gl_LightSource[0].diffuse;

    /* Compute the ambient and globalAmbient terms */
    ambient = gl_FrontColor * gl_LightSource[0].ambient;
    globalAmbient = gl_LightModel.ambient * gl_FrontColor;

    gl_FrontColor =  NdotL * diffuse + globalAmbient + ambient;
    gl_BackColor =  NdotL * diffuse + globalAmbient + ambient;
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

    // number of segments
    const int n = 12;
    // scale factor for the width of the tube
    float scale = gl_FrontColorIn[1].x;
    float stretch1 = gl_FrontColorIn[1].y;
    float stretch2 = gl_FrontColorIn[1].z;

    vec3 ortho1 = normalize( dir20 + dir32);
    vec3 ortho2 = normalize( dir32 + dir53);

    vec3 dir1 = normalize( cross( dir12, ortho1));
    vec3 dir2 = normalize( cross( dir43, ortho2));

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

    dir1 = dir1 * scale;
    dir2 = dir2 * scale;

    for( int i = 0; i < n; i++ )
    {
        alpha = (360.0/float(n))*float(i);
        res1 = ( e + sin( radians( alpha)) * m1 + ( 1.0 - cos( radians( alpha))) * m1quat ) * dir1;
        res2 = ( e + sin( radians( alpha)) * m2 + ( 1.0 - cos( radians( alpha))) * m2quat ) * dir2;

        // copy color
        gl_FrontColor = gl_FrontColorIn[0];
        // copy position
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + res1).xyz, 1.0);
        // compute lighting
        lighting( normalize( res1));
        // done with the vertex
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[3];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + res2).xyz, 1.0);
        lighting( normalize( res2));
        EmitVertex();
    }
    gl_FrontColor = gl_FrontColorIn[0];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1).xyz, 1.0);
    lighting( normalize( dir1));
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[3];
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2).xyz, 1.0);
    lighting( normalize( dir2));
    EmitVertex();

    EndPrimitive();
}
