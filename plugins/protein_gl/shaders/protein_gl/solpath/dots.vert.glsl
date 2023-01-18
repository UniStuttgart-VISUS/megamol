#version 120

uniform vec4 paramSpan; // timeMin, 1/timeRange, speedMin, 1/speedRange
#define TIMEMIN       paramSpan.x
#define TIMERANGEINV  paramSpan.y
#define SPEEDMIN      paramSpan.z
#define SPEEDRANGEINV paramSpan.w

attribute vec2 params; // time, cluster
#define TIME    params.x
#define CLUSTER params.y

void main() {
  gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0f);
  gl_FrontColor = vec4(0.5, 0.75, 1.0, gl_Color.a);
}
