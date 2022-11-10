struct LightParams {
  float px, py, pz, pw;
  float cr, cg, cb, ca;
  float lightIntensity;
};

layout( std430, binding = 2 ) readonly buffer lightBuffer {
  LightParams light[]; 
};

uniform vec4 viewAttr;
uniform vec3 bboxmin;
uniform vec3 bboxmax;
uniform vec3 clipcol;
uniform sampler2D typeData;
uniform ivec2 typeInfo; // type-row, num-faces

uniform vec4 color;
uniform vec4 ambientCol;
uniform vec4 diffuseCol;
uniform vec4 specularCol;
//uniform float lightIntensity;
uniform int numLights;

in vec4 quat;
uniform vec4 camPosInit;
in vec4 camPos;
in vec4 objPos;
//uniform vec4 lightPos;
in float rad;

layout(location = 0) out vec4 outColor;

uniform mat4 ModelViewMatrixInverseTranspose;
uniform mat4 ModelViewProjectionMatrixInverse;
uniform mat4 ModelViewProjectionMatrixTranspose;

in vec4 clipping;
