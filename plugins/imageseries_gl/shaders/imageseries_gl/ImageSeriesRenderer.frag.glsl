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

vec4 labelColor(const float value, const bool label) {
    if ((label && value < (2.0 / 255.0)) || (!label && value < (1.0 / 65535.0)))
        return vec4(0.0, 0.0, 0.0, 1.0);
    if ((!label && (value > (65532.0 / 65535.0))))
        return vec4(1.0, 1.0, 1.0, 1.0);

    const vec4 table[] = vec4[18](
        vec4(0.9, 0.1, 0.1, 1.0),
        vec4(0.1, 0.9, 0.1, 1.0),
        vec4(0.9, 0.9, 0.1, 1.0),
        vec4(0.1, 0.1, 0.9, 1.0),
        vec4(0.9, 0.1, 0.9, 1.0),
        vec4(0.1, 0.9, 0.9, 1.0),
        
        vec4(0.7, 0.1, 0.1, 1.0),
        vec4(0.1, 0.7, 0.1, 1.0),
        vec4(0.7, 0.7, 0.1, 1.0),
        vec4(0.1, 0.1, 0.7, 1.0),
        vec4(0.7, 0.1, 0.7, 1.0),
        vec4(0.1, 0.7, 0.7, 1.0),
        
        vec4(0.9, 0.3, 0.3, 1.0),
        vec4(0.3, 0.9, 0.3, 1.0),
        vec4(0.9, 0.9, 0.3, 1.0),
        vec4(0.3, 0.3, 0.9, 1.0),
        vec4(0.9, 0.3, 0.9, 1.0),
        vec4(0.3, 0.9, 0.9, 1.0));

    const int factor = label ? 255 : 65535;
	const int index = int(floor(value * factor)) % 18;
    
	return table[index];
}

vec4 getColor(vec4 color) {
	switch (displayMode) {
	default:
		return color;
	case 2: // Grayscale
		return color.rrra;
	case 3: // Labels
		return labelColor(color.r, true);
	case 4: // Time index
		return labelColor(color.r, false);
	}
}

void main() {
	fragColor = getColor(texture(image, texCoord));
}
