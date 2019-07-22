in vec4 position;
// in vec4 color;

uniform float inGlobalRadius;
flat out vec4 vsColor;

void main() {
	float radius = (inGlobalRadius>0.0? inGlobalRadius: position.w);
	// float radius = (position.w > 0.0? position.w: inGlobalRadius);
	gl_Position = vec4(position.xyz, radius);
}