#version 430

uniform sampler2D image;
uniform bool grayscale;

in vec2 texCoord;

layout (location = 0) out vec4 fragColor;

void main() {
	vec4 color = texture(image, texCoord);
	if (grayscale) {
	    fragColor = vec4(color.rrr, 1.0);
	} else {
	    fragColor = color;
	}
}
