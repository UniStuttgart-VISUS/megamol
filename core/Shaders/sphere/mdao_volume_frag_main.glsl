out vec4 fsResultColor;
flat in vec4 gsCenterAndSqrRadius;
flat in float gsCellZ;
uniform vec3 inBoundsSizeInverse;
uniform vec3 inVolumeSize;

void main() {

	// The size of one cell in object coordinates
	vec3 cellSize = 1.0/(inVolumeSize * inBoundsSizeInverse);
	
	// Get the min vector of this cell in object coordinates
	vec3 cellMin = vec3(gl_FragCoord.xy - 0.5, gsCellZ)*cellSize;
	
	// Vector from the sphere to the closest cell point
	vec3 sphereCenterToCellMin = cellMin - gsCenterAndSqrRadius.xyz; 
	// Vector from the sphere center to opposite cell point
	vec3 sphereCenterToCellMax = cellMin - gsCenterAndSqrRadius.xyz + cellSize;
	
	// Absolute values of farthest and closest cell corner to the sphere center
	vec3 farthestPoint = max(abs(sphereCenterToCellMin), abs(sphereCenterToCellMax));
	vec3 closestPoint = min(abs(sphereCenterToCellMin), abs(sphereCenterToCellMax));
	
	// Check if the cell is completely outside of the spehre
	// This is the case when the cell corner point with the smallest distance from the sphere
	// center lies outside of the sphere
	if (dot(closestPoint, closestPoint) - gsCenterAndSqrRadius.w > 0.0) {
		discard;
		return;
	}
	
	
	// Check if the cell is completely contained in the sphere
	// This is the case when the cell corner point with the greatest distance from the sphere
	// center lies in the sphere
	if (dot(farthestPoint, farthestPoint) - gsCenterAndSqrRadius.w  < 0.0) {
		fsResultColor = vec4(1.0);
		return;
	}
	
	float radius = sqrt(gsCenterAndSqrRadius.w);
	
	// Check if the sphere is completely contained in the cell
	// This is the case when the bounding box of the sphere is contained in the cell
	if (all(lessThanEqual(sphereCenterToCellMin, -vec3(radius))) && all(greaterThanEqual(sphereCenterToCellMax, vec3(radius)))) {
		// Calculate the volume fraction
		float sphereVol = 4.0/3.0*3.14152 * gsCenterAndSqrRadius.w*radius;
		float cellVol = cellSize.x*cellSize.y*cellSize.z;
		float frac = sphereVol / cellVol;
		fsResultColor = vec4(frac);
		return;
	}
	
	// Now that the farthest point is not within the cell and the sphere is not completely inside the cell we have
	// something in between. We approximate the volume by lame sampling...
	const int samplesPerEdge = 3;
	const int samplesCount = samplesPerEdge * samplesPerEdge * samplesPerEdge;
	const float stepSize = 1.0/(float(samplesPerEdge) - 1.0);
	
	int numHit = 0;
	
	for (int x = 0; x<samplesPerEdge; ++x)
		for (int y=0; y<samplesPerEdge; ++y)
			for (int z=0; z<samplesPerEdge; ++z) {
				// Generate the test point in normalized coordinates
				vec3 sphereCenterToTest = (cellMin + cellSize * vec3(float(x),float(y),float(z)) * stepSize) - gsCenterAndSqrRadius.xyz;

				if (dot(sphereCenterToTest, sphereCenterToTest) - gsCenterAndSqrRadius.w < 0.0)
					numHit++;
			}

	fsResultColor = vec4(float(numHit)/float(samplesCount));
//	fsResultColor = vec4(1.0);
}
