/* Geometry Shader: Cartoon Renderer
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

    // number of segments
    const int n = 6;
    // scale factor for the width of the tube
    float scale = gl_FrontColorIn[1].x;
    float stretch1 = gl_FrontColorIn[1].y;
    float stretch2 = gl_FrontColorIn[1].z;

    vec3 dir1 = normalize( cross( dir12, dir32));
    vec3 dir2 = normalize( cross( dir43, dir53));

    vec3 ortho1 = normalize( dir20 + dir32);
    vec3 ortho2 = normalize( dir32 + dir53);

    // coil/turn
    if( gl_FrontColorIn[2].x < 1.0 )
    {
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

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + res2).xyz, 1.0);
            lighting( normalize( res2));
            EmitVertex();
        }
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1).xyz, 1.0);
        lighting( normalize( dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        EndPrimitive();
/*
        ////////////////
        // draw start //
        ////////////////
        for( int i = 0; i < n; i++ )
        {
            alpha = (360.0/float(n))*float(i);
            res1 = ( e + sin( radians( alpha)) * m2 + ( 1.0 - cos( radians( alpha))) * m2quat ) * dir2;
            alpha = (360.0/float(n))*float(i+1);
            res2 = ( e + sin( radians( alpha)) * m2 + ( 1.0 - cos( radians( alpha))) * m2quat ) * dir2;

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + res1).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + res2).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( v3, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            EndPrimitive();
        }
*/
/*
        //////////////
        // draw end //
        //////////////
        for( int i = 0; i < n; i++ )
        {
            alpha = (360.0/float(n))*float(i);
            res1 = ( e + sin( radians( alpha)) * m1 + ( 1.0 - cos( radians( alpha))) * m1quat ) * dir1;
            alpha = (360.0/float(n))*float(i+1);
            res2 = ( e + sin( radians( alpha)) * m1 + ( 1.0 - cos( radians( alpha))) * m1quat ) * dir1;

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + res1).xyz, 1.0);
            lighting( normalize(-dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + res2).xyz, 1.0);
            lighting( normalize(-dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( v2, 1.0);
            lighting( normalize(-dir20));
            EmitVertex();

            EndPrimitive();
        }
*/
    }
    // sheet ribbon
    else if( gl_FrontColorIn[2].x > 0.5 && gl_FrontColorIn[2].x < 1.5 )
    {
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

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
        lighting( normalize( norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
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

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
        lighting( normalize(-norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
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

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
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

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        EndPrimitive();

        if( gl_FrontColorIn[2].y > 0.0 )
        {
            ////////////////
            // draw start //
            ////////////////
            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
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
    // helix ribbon
    else if( gl_FrontColorIn[2].x > 1.5 && gl_FrontColorIn[2].x < 2.5 )
    {
        dir1 = normalize( cross( dir1, ortho1))*stretch1;
        dir2 = normalize( cross( dir2, ortho2))*stretch2;
        vec3 norm1 = normalize( cross( dir1, ortho1));
        vec3 norm2 = normalize( cross( dir2, ortho2));

        /////////////////////
        // draw top ribbon //
        /////////////////////
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1*0.7 + norm1*scale).xyz, 1.0);
        lighting( normalize( norm1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2*0.7 + norm2*scale).xyz, 1.0);
        lighting( normalize( norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1*0.7 + norm1*scale).xyz, 1.0);
        lighting( normalize( norm1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2*0.7 + norm2*scale).xyz, 1.0);
        lighting( normalize( norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale*0.5).xyz, 1.0);
        lighting( normalize( dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale*0.5).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        EndPrimitive();

        ////////////////////////
        // draw bottom ribbon //
        ////////////////////////
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1*0.7 - norm1*scale).xyz, 1.0);
        lighting( normalize(-norm1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2*0.7 - norm2*scale).xyz, 1.0);
        lighting( normalize(-norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1*0.7 - norm1*scale).xyz, 1.0);
        lighting( normalize(-norm1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2*0.7 - norm2*scale).xyz, 1.0);
        lighting( normalize(-norm2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale*0.5).xyz, 1.0);
        lighting( normalize( dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale*0.5).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        EndPrimitive();

        /////////////////////
        // draw first side //
        /////////////////////
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale*0.5).xyz, 1.0);
        lighting( normalize( dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale*0.5).xyz, 1.0);
        lighting( normalize( dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale*0.5).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale*0.5).xyz, 1.0);
        lighting( normalize( dir2));
        EmitVertex();

        EndPrimitive();

        //////////////////////
        // draw second side //
        //////////////////////
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir1));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale*0.5).xyz, 1.0);
        lighting( normalize(-dir2));
        EmitVertex();

        EndPrimitive();

        if( gl_FrontColorIn[2].y > 0.0 )
        {
            ////////////////
            // draw start //
            ////////////////
            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2*0.7 + norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2*0.7 + norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale*0.5).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale*0.5).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale*0.5).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale*0.5).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2*0.7 - norm2*scale).xyz, 1.0);
            lighting( normalize( dir32));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2*0.7 - norm2*scale).xyz, 1.0);
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
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1*0.7 + norm1*scale).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1*0.7 + norm1*scale).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale*0.5).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale*0.5).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale*0.5).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale*0.5).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1*0.7 - norm1*scale).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            gl_FrontColor = gl_FrontColorIn[0];
            gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1*0.7 - norm1*scale).xyz, 1.0);
            lighting( normalize( -dir20));
            EmitVertex();

            EndPrimitive();
        }
    }
}
