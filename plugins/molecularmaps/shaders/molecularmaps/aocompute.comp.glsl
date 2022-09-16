layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

struct ownVector 
{
    float x;
    float y;
    float z;
};

layout(std430, binding = 1) buffer VertexList
{
    ownVector vertices[];
};

layout(std430, binding = 2) buffer NormalList
{
    ownVector normals[];
};

layout(std430, binding = 3) buffer ColourList
{
    vec4 outColours[];
};

uniform float aoSampFact = 1.0; // factor for AO samples
uniform int vertexCount; // number of vertices in the vertex list
uniform int sampleNum = 8; // number of samples per vertex
uniform int sampleMax; // number of volume samples per direction
uniform vec3 posOrigin; // origin of the bounding box
uniform vec3 posExtents; // measurements of the bounding box

vec3 bbMin = posOrigin;
vec3 bbMax = posOrigin + posExtents;

// necessary textures
uniform sampler1D directionTex;
uniform sampler1D levelTex;
uniform sampler3D aoVol;

// generates a rotation matrix that rotates the z-axis onto the given normal
mat4 computeRotMatrix(vec3 n) {
    vec3 zdir = vec3(0,0,1);
    vec3 around = cross(zdir, n);
    float angle = acos(dot(zdir, n));
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0f - cos(angle);

    mat4 rot = mat4(1.0);
    rot[0][0] = t * (around.x * around.x) + c;
    rot[1][0] = t * (around.x * around.y) - s * around.z;
    rot[2][0] = t * (around.x * around.z) + s * around.y;
    
    rot[0][1] = t * (around.y * around.x) + s * around.z;
    rot[1][1] = t * (around.y * around.y) + c;
    rot[2][1] = t * (around.y * around.z) - s * around.x;
    
    rot[0][2] = t * (around.x * around.z) - s * around.y;
    rot[1][2] = t * (around.y * around.z) + s * around.x;
    rot[2][2] = t * (around.z * around.z) + c;

    return rot;
}

float distFalloff(float dist, float maxdist) {
    return dist / maxdist;
}

// intersects the given ray with the bounding box
// and returns the distances of the two intersection points
// The returned value is the distance of the first hit point to the origin
// if it is negative, there is no hit
// https://www.gamedev.net/topic/682750-problem-with-raybox-intersection-in-glsl/
float intersectBoundingBox(vec3 origin, vec3 direction) {
    vec3 v1 = (bbMin - origin) * (-direction);
    vec3 v2 = (bbMax - origin) * (-direction);
    vec3 n = min(v1, v2);
    vec3 f = max(v1, v2);
    float enter = max(n.x, max(n.y, n.z));
    float ex = min(f.x, min(f.y, f.z));
    return max(ex, enter); // TODO is this correct?
}

void main(void) {
    uint u = gl_GlobalInvocationID.x;
    
    // stop the computation if we have reached the index of a non-existing vertex
    if (u >= vertexCount) {
        return;
    }

    float aoFactor = 0.0;
    vec3 aoPos = vec3(vertices[u].x, vertices[u].y, vertices[u].z);
    vec3 normal = normalize(vec3(normals[u].x, normals[u].y, normals[u].z));
    //normal = -normal;

    mat4 rotMat = computeRotMatrix(normal);

    float divisor = (posExtents.x + posExtents.y + posExtents.z) / 3.0;
    divisor /= 10.0;

    int directionalHits = 0;

    // hemisphere sampling
    for (int i = 0; i < sampleNum; i++) {
        float density = 0.0;

        // we have to rotate the main direction onto the surface normal
        vec4 dir = texture(directionTex, (float(i) + 0.5) / (float(sampleNum))).xyzw;
        vec3 dirT = vec4(rotMat * vec4(dir.xyz, 1.0)).xyz;
        dirT = normalize(dirT);

        float distBox = intersectBoundingBox(aoPos, dirT);
        if(distBox < 0.0) continue;

        vec4 levelPara = texture(levelTex, 0.0).xyzw;
        int numSamples = int(distBox / levelPara.y);
        float dist = 0.0;
        dist = 1.0;
        float texVal;

        int numHits = 0;

        // sample along the ray
        for(int j = 0; j < numSamples; j++) {
            dist += levelPara.y;
            vec3 pos = aoPos + dirT * dist;
            pos = (pos - posOrigin) / posExtents;
            texVal = texture(aoVol, pos).r * distFalloff(dist, divisor);

            if(texVal > 0.6) {
                numHits += 1;
            }
        }

        if(numHits > 0) {
            directionalHits += 1;
        }
        aoFactor = distBox;
    }
    aoFactor = float(directionalHits) / float(sampleNum);

    //aoFactor *= aoSampFact;
    aoFactor = 1.0 - aoFactor;
    outColours[u] = vec4(aoFactor);
    vec4 bla = texture(directionTex, 0.0).xyzw;
    vec4 bla2 = vec4(rotMat * vec4(bla.xyz, 1.0));
    //outColours[u] = vec4(bla2.xyz, 1.0);
    //outColours[u] = vec4(normal, 1.0);
}
