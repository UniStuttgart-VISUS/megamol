in vec3 inPosition;
in vec4 inColor;
in vec2 inTexture;
in vec4 inAttributes;

uniform vec2 viewport;
uniform mat4 mvp;

flat out vec4 color;
flat out vec2 center;
flat out float radius;
out vec2 texcoord;
flat out vec4 attributes;
