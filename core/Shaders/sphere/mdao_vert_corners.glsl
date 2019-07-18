vec2 d, p, q, h, dd;

// get camera orthonormal coordinate system
vec4 tmp;

vec2 mins, maxs;
vec3 testPos;
vec4 projPos;

// projected camera vector
vec3 c2 = vec3(dot(vsCamPos.xyz, inCamRight), dot(vsCamPos.xyz, inCamUp), dot(vsCamPos.xyz, inCamFront));

vec3 cpj1 = inCamFront * c2.z + inCamRight * c2.x;
vec3 cpm1 = inCamFront * c2.x - inCamRight * c2.z;

vec3 cpj2 = inCamFront * c2.z + inCamUp * c2.y;
vec3 cpm2 = inCamFront * c2.y - inCamUp * c2.z;

d.x = length(cpj1);
d.y = length(cpj2);

dd = vec2(1.0) / d;

p = vsSquaredRad * dd;
q = d - p;
h = sqrt(p * q);

p *= dd;
h *= dd;

cpj1 *= p.x;
cpm1 *= h.x;
cpj2 *= p.y;
cpm2 *= h.y;

testPos = vsObjPos.xyz + cpj1 + cpm1;
projPos = inMvp * vec4(testPos, 1.0);
projPos /= projPos.w;
mins = projPos.xy;
maxs = projPos.xy;

testPos -= 2.0 * cpm1;
projPos = inMvp * vec4(testPos, 1.0);
projPos /= projPos.w;
mins = min(mins, projPos.xy);
maxs = max(maxs, projPos.xy);

testPos = vsObjPos.xyz + cpj2 + cpm2;
projPos = inMvp * vec4(testPos, 1.0);
projPos /= projPos.w;
mins = min(mins, projPos.xy);
maxs = max(maxs, projPos.xy);

testPos -= 2.0 * cpm2;
projPos = inMvp * vec4(testPos, 1.0);
projPos /= projPos.w;
mins = min(mins, projPos.xy);
maxs = max(maxs, projPos.xy);