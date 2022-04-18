#version 430

uniform sampler2D image;
uniform int displayMode;

in vec2 texCoord;

layout (location = 0) out vec4 fragColor;

vec4 twoWayGradient(float factor, vec4 minColor, vec4 midColor, vec4 maxColor) {
	return mix(
		mix(midColor, minColor, min(1, -factor)),
		mix(midColor, maxColor, min(1, factor)),
		step(0, factor));
}

vec4 timeDifference(vec4 color) {
	float weight = color.r * 2.0 - 1.0;
	float uncertainty = 1.0 - color.g;

	vec4 minColor = vec4(0.0, 0.1, 1.0, 1.0);
	vec4 midColor = mix(vec4(1.0), vec4(vec3(0.6), 1.0), uncertainty);
	vec4 maxColor = vec4(1.0, 0.1, 0.0, 1.0);

	return twoWayGradient(weight, minColor, midColor, maxColor);
}

vec3 hsv(vec3 color)
{
    vec3 p = clamp(abs(fract(color.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3(3.0)) - vec3(1.0), 0.0, 1.0);
    return color.z * mix(vec3(1.0), p, color.y);
}

vec4 labelColor(vec4 color) {
	float label = floor(color.r * 255.0) + 1.0;
    float ring = floor(log2(max(label, 1.0))) - 1.0;
    float subValue = label / (exp2(ring + 1.1));

	vec3 rainbow = hsv(vec3(subValue, 1.0 / max(1.0, ring), 1.0 - ring * 0.05));
	vec3 gray = hsv(vec3(0.1, 0.1, (label - 1.0) / 3.0));

	return vec4(mix(rainbow, gray, step(label, 3.0)), color.a);
}

vec4 getColor(vec4 color) {
	switch (displayMode) {
	default:
		return color;
	case 2: // Grayscale
		return color.rrra;
	case 3: // Labels
		return labelColor(color);
	case 4: // Time diff
		return timeDifference(color);
	}
}

void main() {
	fragColor = getColor(texture(image, texCoord));
}
