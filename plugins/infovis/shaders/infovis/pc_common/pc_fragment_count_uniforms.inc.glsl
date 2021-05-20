layout(binding = 0) uniform sampler1D transferFunction;
layout(binding = 1) uniform sampler2D fragmentCount;

uniform vec2 scaling = vec2(1.0);
uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

uniform vec4 clearColor = vec4(0.0);

uniform uint invocationCount = 0;
uniform int sqrtDensity = 1;
