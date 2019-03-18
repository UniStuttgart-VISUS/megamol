#define CLIP
#define DEPTH

uniform vec4 inViewAttr;

in vec4 vsObjPos;
in vec4 vsCamPos;
in float vsSquaredRad;
in float vsRad;

in vec4 vsColor;

uniform mat4 inMvpInverse;
uniform mat4 inMvpTrans;

uniform vec4 inClipDat;
uniform vec4 inClipCol;

uniform bool inUseHighPrecision;

out vec4 outColor;
out vec4 outNormal;


void main(void) {

    vec4 coord;
    vec3 ray;
    float lambda;

	// transform fragment coordinates from window coordinates to view coordinates.
	coord = gl_FragCoord 
		* vec4(inViewAttr.z, inViewAttr.w, 2.0, 0.0)
		+ vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = inMvpInverse * coord;
    coord /= coord.w;
    coord -= vsObjPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - vsCamPos.xyz);

    // chose color for lighting
    vec4 color = vsColor;
	
    // calculate the geometry-ray-intersection
	float cam_dot_ray = dot(vsCamPos.xyz, ray);
	float cam_dot_cam = dot(vsCamPos.xyz, vsCamPos.xyz);
	
	// off axis of cam-sphere-vector and ray
    float d2s = cam_dot_cam - cam_dot_ray * cam_dot_ray;      
    float radicand = vsSquaredRad - d2s;                        // square of difference of projected length and lambda
    if (radicand < 0.0) {
        discard; 
    }

    float sqrtRadicand = sqrt(radicand);
    lambda = -cam_dot_ray - sqrtRadicand;                           // lambda

    vec3 sphereintersection = lambda * ray + vsCamPos.xyz;    // intersection point in camera coordinates
    vec4 normal = vec4(sphereintersection / vsRad, 0.0);
	
	if (any(notEqual(inClipDat.xyz, vec3(0.0, 0.0, 0.0)))) {
		vec3 planeNormal = normalize(inClipDat.xyz);
		vec3 clipPlaneBase = planeNormal * inClipDat.w;
		float d = -dot(planeNormal, clipPlaneBase - vsObjPos.xyz);
		float dist1 = dot(sphereintersection, planeNormal) + d;
		float dist2 = d;
		float t = -(dot(planeNormal, vsCamPos.xyz) + d) / dot(planeNormal, ray);
		vec3 planeintersect = vsCamPos.xyz + t * ray;
		if (dist1 > 0.0) {
			if (dist2*dist2 < vsSquaredRad) {
				if (dot(planeintersect, planeintersect) < vsSquaredRad) {
					sphereintersection = planeintersect;
					normal = vec4(planeNormal, 1.0);
					color = mix(color, vec4(inClipCol.rgb, 1.0), inClipCol.a);
				} else {
					discard;
				}
			} else {
				discard;
			}
		}
	} 

	outColor = vec4(color.rgb, 1.0);
	outNormal = normal;
	if (!inUseHighPrecision)
		outNormal = outNormal * 0.5 + 0.5;
	
	vec4 hitPosObj = vec4(sphereintersection + vsObjPos.xyz, 1.0);
    float depth = dot(inMvpTrans[2], hitPosObj);
    float depthW = dot(inMvpTrans[3], hitPosObj);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
}