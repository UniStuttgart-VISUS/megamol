
	// transform fragment coordinates from window coordinates to view coordinates.
	coord = gl_FragCoord 
		* vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
		+ vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // chose color for lighting
    vec4 color = vertColor;
	
    // calculate the geometry-ray-intersection
	float cam_dot_ray = dot(camPos.xyz, ray);
	float cam_dot_cam = dot(camPos.xyz, camPos.xyz);
	
	// off axis of cam-sphere-vector and ray
    float d2s = cam_dot_cam - cam_dot_ray * cam_dot_ray;      
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
    if (radicand < 0.0) {
        discard; 
    }

    float sqrtRadicand = sqrt(radicand);
    lambda = -cam_dot_ray - sqrtRadicand;                           // lambda

    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point in camera coordinates
    vec4 normal = vec4(sphereintersection / rad, 0.0);
	
	if (any(notEqual(clipDat.xyz, vec3(0.0, 0.0, 0.0)))) {
		vec3 planeNormal = normalize(clipDat.xyz);
		vec3 clipPlaneBase = planeNormal * clipDat.w;
		float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
		float dist1 = dot(sphereintersection, planeNormal) + d;
		float dist2 = d;
		float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
		vec3 planeintersect = camPos.xyz + t * ray;
		if (dist1 > 0.0) {
			if (dist2*dist2 < squarRad) {
				if (dot(planeintersect, planeintersect) < squarRad) {
					sphereintersection = planeintersect;
					normal = vec4(planeNormal, 1.0);
					color = mix(color, vec4(clipCol.rgb, 1.0), clipCol.a);
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
	
	vec4 hitPosObj = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], hitPosObj);
    float depthW = dot(MVPtransp[3], hitPosObj);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
}
