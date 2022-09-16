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

uniform vec4 color;
uniform vec4 ambientCol;
uniform vec4 diffuseCol;
uniform vec4 specularCol;
//uniform float lightIntensity;
uniform int numLights;

varying vec4 quat;
uniform vec4 camPosInit;
varying vec4 camPos;
varying vec4 objPos;
//uniform vec4 lightPos;
varying float rad;

uniform mat4 ModelViewMatrixInverseTranspose;
uniform mat4 ModelViewProjectionMatrixInverse;
uniform mat4 ModelViewProjectionMatrixTranspose;

varying vec4 clipping;
