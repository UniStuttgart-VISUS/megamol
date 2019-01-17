layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 modelview;
uniform mat4 proj;
uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)
uniform vec4 lightPos;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

in vec4 inColor[1];

out vec4 color;
out vec4 objPos;
out vec4 camPos;
out vec4 light;
out float rad;
out float squareRad;

mat4 modelviewproj = proj*modelview; // TODO Move this to the CPU?
mat4 modelviewInv = inverse(modelview);

void main(void) {
    
    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_in[0].gl_Position;
    rad = inPos.w;
    squareRad = rad*rad;
    inPos.w = 1.0;

    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = modelviewInv[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space
    
    // calculate light position in glyph space
    // USE THIS LINE TO GET POSITIONAL LIGHTING
    //lightPos = modelviewInv * gl_LightSource[0].position - objPos;
    // USE THIS LINE TO GET DIRECTIONAL LIGHTING
    light = modelviewInv*normalize(lightPos);
    
    color = inColor[0];    
    
       // Sphere-Touch-Plane-Approach™
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

/*#ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = gl_ModelViewMatrixInverse[3] + gl_ModelViewMatrixInverse[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = gl_ModelViewMatrixInverse[3] + gl_ModelViewMatrixInverse[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = cross(camIn, camRight);
#endif // CALC_CAM_SYS*/

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    
#ifdef HALO
    squarRad = (rad + HALO_RAD) * (rad + HALO_RAD);
#endif // HALO
    
    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));

    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;

    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;
    
    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    p = squareRad * dd;
    q = d - p;
    h = sqrt(p * q);
    //h = vec2(0.0);
    
    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = modelviewproj * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = modelviewproj * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = modelviewproj * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = modelviewproj * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    //gl_Position = vec4((mins + maxs) * 0.5, 0.0, /*1.0*/inPos.w);
    //gl_PointSize = max((maxs.x - mins.x) * winHalf.x, (maxs.y - mins.y) * winHalf.y) * 0.5;
    
    // Cube vertices
    /*vec4 posA =  modelviewproj * vec4(objPos.xyz + (camRight + camUp + camIn)*rad, 1.0);
    vec4 posB =  modelviewproj * vec4(objPos.xyz + (camRight - camUp + camIn)*rad, 1.0);
    vec4 posC =  modelviewproj * vec4(objPos.xyz + (-camRight + camUp + camIn)*rad, 1.0);
    vec4 posD =  modelviewproj * vec4(objPos.xyz + (-camRight - camUp + camIn)*rad, 1.0);
    vec4 posE =  modelviewproj * vec4(objPos.xyz + (-camRight - camUp - camIn)*rad, 1.0);
    vec4 posF =  modelviewproj * vec4(objPos.xyz + (camRight - camUp - camIn)*rad, 1.0);
    vec4 posG =  modelviewproj * vec4(objPos.xyz + (camRight + camUp - camIn)*rad, 1.0);
    vec4 posH =  modelviewproj * vec4(objPos.xyz + (-camRight + camUp - camIn)*rad, 1.0);*/
    
    // Triangle strip
    /*gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posD; EmitVertex();
    gl_Position = posE; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posF; EmitVertex();
    gl_Position = posG; EmitVertex();
    gl_Position = posE; EmitVertex();
    gl_Position = posH; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posG; EmitVertex();
    gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posD; EmitVertex();
    gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();*/
    gl_Position = vec4(mins.x, maxs.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(mins.x, mins.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(maxs.x, maxs.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(maxs.x, mins.y, 0.0, inPos.w); EmitVertex();
    EndPrimitive();
}