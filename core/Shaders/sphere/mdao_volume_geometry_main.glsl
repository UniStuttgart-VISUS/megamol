#extension GL_EXT_geometry_shader4: enable

layout (points) in;
layout (points, max_vertices = 64) out;

uniform vec3 inBoundsMin;
uniform vec3 inBoundsSizeInverse;
uniform vec3 inVolumeSize;
uniform float inGlobalRadius;

uniform vec4 clipDat;

flat out vec4 gsCenterAndSqrRadius;
flat out float gsCellZ;

vec3 normalizedPosition;

// Emit a disk by setting a position and a layer (to be rendered into the volume texture)
// Also set the point size to serve as a canvas and pass the fragment shader a vector
// that contains the disk center (in moved object coordinates) and the squared radius
void emitDisk(int layer, float diskSize, float radius) {

	gl_Layer = layer;
	gsCellZ = layer;
	gl_PointSize = int(2.0*ceil(diskSize * max(inBoundsSizeInverse.x*inVolumeSize.x, inBoundsSizeInverse.y*inVolumeSize.y)));
	gl_Position = vec4(normalizedPosition.xyz*2.0 - 1.0, 1.0);
	gsCenterAndSqrRadius = vec4(gl_PositionIn[0].xyz - inBoundsMin, radius*radius);

	EmitVertex();
	EndPrimitive();
}

// Create information for a whole hemisphere (excluding the center cross section) by
// calculating radii of cross sections and emitting a disk of appropriate size
void emitHemisphere(int layer, int numSlices, float radius, float offset, int sliceStep) {

	for (int i=1; i<=numSlices; ++i) {
		// Calculate the distance to the nearest border of the current slice (in object coordinates)
		
		float dist = (float(i-1)/inVolumeSize.z + offset) / inBoundsSizeInverse.z;
		
		//float dist = ((float(i-1)-0.5)/inVolumeSize.z + offset) / inBoundsSizeInverse.z;
		// Recalculate the new radius
		float newSqrRadius = radius*radius - dist*dist;

		// FIXME: Can this happen?
		if (newSqrRadius < 0.0)
			break;
		
		// Emit a new disk
		emitDisk(layer + sliceStep * i, sqrt(newSqrRadius), radius);
	} 

}



void main() {

    // clipping
    float od = clipDat.w - 1.0;
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
		od = dot(gl_PositionIn[0].xyz, clipDat.xyz) - gl_PositionIn[0].w;
	}
	if (od > clipDat.w)
		return;
		
	// Convert object space to normalized pixel space in [0 - 1]
	normalizedPosition = (gl_PositionIn[0].xyz - inBoundsMin)*inBoundsSizeInverse;

	float radius = gl_PositionIn[0].w;
	
	// Calculate the number of slices needed for one half of the sphere
	int numSlices = int(ceil(radius * inBoundsSizeInverse.z * inVolumeSize.z));

	gl_Position = vec4(normalizedPosition.xyz*2.0 - 1.0, 1.0);

	// Calculate the center layer
	int centerLayer = int(floor(normalizedPosition.z * inVolumeSize.z/* + 0.5*/));
	emitDisk(centerLayer, radius, radius);
	
	// Calculate the offset in Z direction from the center of the center layer
	float sliceThickness = 1.0 / inVolumeSize.z; 
	float zOffset = fract(normalizedPosition.z * inVolumeSize.z)/inVolumeSize.z;

	// Emit a hemisphere in negative direction
	emitHemisphere(centerLayer, numSlices, radius, zOffset, -1);
	// Emit a hemisphere in positive direction
	emitHemisphere(centerLayer, numSlices, radius, sliceThickness - zOffset, +1);
}