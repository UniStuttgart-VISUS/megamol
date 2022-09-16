uniform vec4 paramSpan; // timeMin, 1/timeRange, speedMin, 1/speedRange
#define TIMEMIN       paramSpan.x
#define TIMERANGEINV  paramSpan.y
#define SPEEDMIN      paramSpan.z
#define SPEEDRANGEINV paramSpan.w

attribute vec2 params; // time, cluster
#define TIME    params.x
#define CLUSTER params.y

varying vec2 values; // time, speed
#define OUTTIME  values.x
#define OUTSPEED values.y

void main() {
  gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0f);

  OUTTIME = clamp((TIME - TIMEMIN) * TIMERANGEINV, 0.0, 1.0);
  OUTSPEED = clamp((gl_Vertex.w - SPEEDMIN) * SPEEDRANGEINV, 0.0, 1.0);
}
