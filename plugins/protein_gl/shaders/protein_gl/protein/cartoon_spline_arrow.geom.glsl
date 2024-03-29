/* Geometry Shader: Arrow Renderer
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

varying out vec3 normal;
varying out vec3 color;

void compute( vec3 v2, vec3 v3, vec3 dir20, vec3 dir12, vec3 dir32, vec3 dir43, vec3 dir53, float scale, float stretch1, float stretch2)
{
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
    color = vec3( gl_FrontSecondaryColorIn[0]);
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale).xyz, 1.0);
    normal = norm1;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale).xyz, 1.0);
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
    normal = norm2;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
    EmitVertex();

    EndPrimitive();

    ////////////////////////
    // draw bottom ribbon //
    ////////////////////////
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale).xyz, 1.0);
    normal = -norm1;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale).xyz, 1.0);
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
    normal = -norm2;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale).xyz, 1.0);
    EmitVertex();

    EndPrimitive();

    /////////////////////
    // draw first side //
    /////////////////////
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 + norm1*scale).xyz, 1.0);
    normal = dir1;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 + dir1 - norm1*scale).xyz, 1.0);
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 + norm2*scale).xyz, 1.0);
    normal = dir2;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 + dir2 - norm2*scale).xyz, 1.0);
    EmitVertex();

    EndPrimitive();

    //////////////////////
    // draw second side //
    //////////////////////
    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 + norm1*scale).xyz, 1.0);
    normal = -dir1;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v2 - dir1 - norm1*scale).xyz, 1.0);
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 + norm2*scale).xyz, 1.0);
    normal = -dir2;
    EmitVertex();

    gl_Position = gl_ModelViewProjectionMatrix*vec4( (v3 - dir2 - norm2*scale).xyz, 1.0);
    EmitVertex();

    EndPrimitive();
}

void main(void)
{
    // number of segments for each curve section
    const int numSeg = 6;
    const float N = float( numSeg);
    // dimension of 'results' and 'directions' MUST be numSeg+1
    vec3 results[numSeg + 1];
    vec3 directions[numSeg + 1];
    // vertices and directions
    // vertices and directions
    vec3 v1 = vec3(gl_PositionIn[0].xyz) / gl_PositionIn[0].w;
    vec3 v2 = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;
    vec3 v3 = vec3(gl_PositionIn[2].xyz) / gl_PositionIn[2].w;
    vec3 v4 = vec3(gl_PositionIn[3].xyz) / gl_PositionIn[3].w;
    vec3 n1 = vec3(gl_FrontColorIn[0].xyz) / gl_FrontColorIn[0].w;
    vec3 n2 = vec3(gl_FrontColorIn[1].xyz) / gl_FrontColorIn[1].w;
    vec3 n3 = vec3(gl_FrontColorIn[2].xyz) / gl_FrontColorIn[2].w;
    vec3 n4 = vec3(gl_FrontColorIn[3].xyz) / gl_FrontColorIn[3].w;

    // assign matrix S
  mat4 S;
    S[0] = vec4( 6.0/(N*N*N), 6.0/(N*N*N), 1.0/(N*N*N), 0.0 );
    S[1] = vec4(         0.0,   2.0/(N*N),   1.0/(N*N), 0.0 );
    S[2] = vec4(                 0.0,         0.0,       1.0/N, 0.0 );
    S[3] = vec4(                 0.0,         0.0,         0.0, 1.0 );
    // assign the basis matrix
  mat4 B;
    B[0] = vec4(-1.0/6.0, 3.0/6.0,-3.0/6.0, 1.0/6.0 );
    B[1] = vec4( 3.0/6.0,-6.0/6.0, 0.0/6.0, 4.0/6.0 );
    B[2] = vec4(-3.0/6.0, 3.0/6.0, 3.0/6.0, 1.0/6.0 );
    B[3] = vec4( 1.0/6.0, 0.0/6.0, 0.0/6.0, 0.0/6.0 );

    // assign the geometry matrix
    mat4 G;
    G[0] = vec4( v1.x, v2.x, v3.x, v4.x );
    G[1] = vec4( v1.y, v2.y, v3.y, v4.y );
    G[2] = vec4( v1.z, v2.z, v3.z, v4.z );
    G[3] = vec4( 1.0, 1.0, 1.0, 1.0 );
    // compute the matrix M
    mat4 M = S * ( B * G );
    // start computation of first spline segment
    results[0] = vec3( M[0][3]/M[3][3], M[1][3]/M[3][3], M[2][3]/M[3][3]);
    for( int k = 0; k < numSeg; k++ )
    {
        M[0].w = M[0].w + M[0].z;
        M[1].w = M[1].w + M[1].z;
        M[2].w = M[2].w + M[2].z;
        M[3].w = M[3].w + M[3].z;
        M[0].z = M[0].z + M[0].y;
        M[1].z = M[1].z + M[1].y;
        M[2].z = M[2].z + M[2].y;
        M[3].z = M[3].z + M[3].y;
        M[0].y = M[0].y + M[0].x;
        M[1].y = M[1].y + M[1].x;
        M[2].y = M[2].y + M[2].x;
        M[3].y = M[3].y + M[3].x;

        results[k+1] = vec3( M[0][3]/M[3][3], M[1][3]/M[3][3], M[2][3]/M[3][3]);
    }

    mat4 G1;
    G1[0] = vec4( n1.x+v1.x, n2.x+v2.x, n3.x+v3.x, n4.x+v4.x );
    G1[1] = vec4( n1.y+v1.y, n2.y+v2.y, n3.y+v3.y, n4.y+v4.y );
    G1[2] = vec4( n1.z+v1.z, n2.z+v2.z, n3.z+v3.z, n4.z+v4.z );
    G1[3] = vec4( 1.0, 1.0, 1.0, 1.0 );

    // compute the matrix M
    mat4 M1 = S * ( B * G1 );
    // start computation of second spline segments
    directions[0] = vec3( M1[0][3]/M1[3][3], M1[1][3]/M1[3][3], M1[2][3]/M1[3][3]);
    for( int k = 0; k < numSeg; k++ )
    {
        M1[0].w = M1[0].w + M1[0].z;
        M1[1].w = M1[1].w + M1[1].z;
        M1[2].w = M1[2].w + M1[2].z;
        M1[3].w = M1[3].w + M1[3].z;
        M1[0].z = M1[0].z + M1[0].y;
        M1[1].z = M1[1].z + M1[1].y;
        M1[2].z = M1[2].z + M1[2].y;
        M1[3].z = M1[3].z + M1[3].y;
        M1[0].y = M1[0].y + M1[0].x;
        M1[1].y = M1[1].y + M1[1].x;
        M1[2].y = M1[2].y + M1[2].x;
        M1[3].y = M1[3].y + M1[3].x;

        directions[k+1] = vec3( M1[0][3]/M1[3][3], M1[1][3]/M1[3][3], M1[2][3]/M1[3][3]);
    }

    // draw
    vec3 dir20;
    vec3 dir12;
    vec3 dir32;
    vec3 dir43;
    vec3 dir53;

    float factor = ( gl_FrontSecondaryColorIn[1].y / N ) * gl_FrontSecondaryColorIn[1].z;
    float f2;
    float f1 = gl_FrontSecondaryColorIn[1].y + factor;

    f2 = f1;
    f1 = f1 - factor;
    // compute all needed directions
    dir20 = v3 - v1;
    dir12 = directions[0] - results[0];
    dir32 = results[1] - results[0];
    dir43 = directions[1] - results[1];
    dir53 = results[2] - results[1];
    compute( results[0], results[1], dir20, dir12, dir32, dir43, dir53, gl_FrontSecondaryColorIn[1].x, f2, f1 );

    for( int i = 0; i < numSeg-2; i++ )
    {
        f2 = f1;
        f1 = f1 - factor;
        // compute all needed directions
        dir20 = results[i+1] - results[i+0];
        dir12 = directions[i+1] - results[i+1];
        dir32 = results[i+2] - results[i+1];
        dir43 = directions[i+2] - results[i+2];
        dir53 = results[i+3] - results[i+2];
        compute( results[i+1], results[i+2], dir20, dir12, dir32, dir43, dir53, gl_FrontSecondaryColorIn[1].x, f2, f1 );
    }

    f2 = f1;
    f1 = f1 - factor;
    // compute all needed directions
    dir20 = results[numSeg-1] - results[numSeg-2];
    dir12 = directions[numSeg-1] - results[numSeg-1];
    dir32 = results[numSeg] - results[numSeg-1];
    dir43 = directions[numSeg] - results[numSeg];
    dir53 = v4 - v2;
    compute( results[numSeg-1], results[numSeg], dir20, dir12, dir32, dir43, dir53, gl_FrontSecondaryColorIn[1].x, f2, f1 );
}
