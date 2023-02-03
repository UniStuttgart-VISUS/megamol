#version 430

in vec2 texCoord;

out vec4 fragColor;

void main() {
    float radius = sqrt(texCoord.x * texCoord.x + texCoord.y * texCoord.y);

    float fadeout_region = 0.95;

    float alpha = 1.0;
    if (radius > 1.0) alpha = 0.0;
    else if (radius > fadeout_region) alpha = 1.0 - ((radius - fadeout_region) / (1.0 - fadeout_region));

    fragColor = vec4(1.0, 1.0, 1.0, alpha);
}
