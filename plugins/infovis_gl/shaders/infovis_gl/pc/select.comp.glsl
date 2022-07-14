#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"

#ifdef STROKE
uniform vec2 strokeStart = vec2(0.0f, 0.0f);
uniform vec2 strokeEnd = vec2(0.0f, 0.0f);
#else
uniform vec2 mouse = vec2(0.0f, 0.0f);
uniform float pickRadius = 0.1f;
#endif

#define FLOAT_EPS (1.0e-10)

bool intersectLineCircle(vec2 p, vec2 q, vec2 m, float r) {
    // Project m onto (p, q)

    vec2 x = m - p;
    vec2 l = q - p;

    float lineLength = dot(l, l);

    if (abs(lineLength) < FLOAT_EPS) {
        return false;
    }

    float u = dot(x, l) / lineLength;

    if (u < 0.0) {
        // x is already correct
    } else if (u > 1.0) {
        x = m - q;
    } else { // 0.0 < u < 1.0
        x -= u * l;
    }

    return dot(x, x) <= (r * r);
}

// @see http://stackoverflow.com/a/565282/791895
float cross2(vec2 v, vec2 w) {
    return v.x * w.y - v.y * w.x;
}

bool intersectLineLine(vec2 p, vec2 r, vec2 q, vec2 s) {
    float rXs = cross2(r, s);

    if (abs(rXs) > FLOAT_EPS) {
        vec2 qp = q - p;
        float t = cross2(qp, s) / rXs;
        float u = cross2(qp, r) / rXs;

        return (0.0 <= t) && (t <= 1.0) && (0.0 <= u) && (u <= 1.0);
    }

    return false;
}

void main() {
    uint itemIdx = globalInvocationIndex();

    if (itemIdx >= itemCount || !bitflag_isVisible(flags[itemIdx])) {
        return;
    }

    bool selected = false;

    for (uint axisIdx = 1; axisIdx < dimensionCount; axisIdx++) {
        vec4 p = pc_itemVertex(itemIdx, axisIdx - 1);
        vec4 q = pc_itemVertex(itemIdx, axisIdx);

#ifdef STROKE
        if (intersectLineLine(strokeStart, strokeEnd - strokeStart, p.xy, q.xy - p.xy)) {
#else
        if (intersectLineCircle(p.xy, q.xy, mouse, pickRadius)) {
#endif
            selected = true;
            break;
        }
    }

    bitflag_set(flags[itemIdx], FLAG_SELECTED, selected);
}
