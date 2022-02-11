#version 430

uniform sampler2D image;
uniform bool grayscale;

in vec2 texCoord;

layout (location = 0) out vec4 fragColor;

vec4 twoWayGradient(float factor, vec4 minColor, vec4 midColor, vec4 maxColor) {
	return mix(
		mix(midColor, minColor, min(1, -factor)),
		mix(midColor, maxColor, min(1, factor)),
		step(0, factor));
}

void main() {
	vec4 color = texture(image, texCoord);
	if (grayscale) {
	    fragColor = vec4(color.rrr, 1.0);
	} else {
		// TODO add this as a dedicated renderer mode
	    //fragColor = color;
		float weight = color.r * 2.0 - 1.0;
		float uncertainty = 1.0 - color.g;

		vec4 minColor = vec4(0.0, 0.1, 1.0, 1.0);
		vec4 midColor = mix(vec4(1.0), vec4(vec3(0.6), 1.0), uncertainty);
		vec4 maxColor = vec4(1.0, 0.1, 0.0, 1.0);

		fragColor = twoWayGradient(weight, minColor, midColor, maxColor);


	}
}
