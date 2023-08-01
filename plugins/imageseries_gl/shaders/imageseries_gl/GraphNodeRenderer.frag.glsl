#version 430

layout(location = 0) in vec2 texCoord;
layout(location = 1) in float nodeType;

out vec4 fragColor;

bool is_approx(float ref, float val) {
    return val < ref + 0.1 && val > ref - 0.1;
}

void main() {
    float radius = sqrt(texCoord.x * texCoord.x + texCoord.y * texCoord.y);

    float fadeout_region = 0.99;

    float alpha = 1.0;
    if (radius > 1.0) alpha = 0.0;
    else if (radius > fadeout_region) alpha = 1.0 - ((radius - fadeout_region) / (1.0 - fadeout_region));

    vec3 color = vec3(1.0, 1.0, 1.0);
    if (is_approx(6.0, nodeType)) { // isolated/invalid
        alpha = 0.0;
        color = vec3(1.0, 1.0, 0.0);
    } else if (is_approx(5.0, nodeType)) { // multi
        color = vec3(0.784, 0.392, 0.784);
    } else if (is_approx(4.0, nodeType)) { // merge
        color = vec3(1.0, 0.627, 0.392);
    } else if (is_approx(3.0, nodeType)) { // split
        color = vec3(0.392, 1.0, 1.0);
    } else if (is_approx(2.0, nodeType)) { // sink
        color = vec3(1.0, 0.235, 0.235);
    } else if (is_approx(1.0, nodeType)) { // source
        color = vec3(0.235, 1.0, 0.235);
    }

    fragColor = vec4(color, alpha);
}
