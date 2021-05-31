#define FLOAT_EPS (1.0e-10)

bool intersectLineCircle(vec2 p, vec2 q, vec2 m, float r)
{
    // Project m onto (p, q)

    vec2 x = m - p;
    vec2 l = q - p;

    float lineLength = dot(l, l);

    if (abs(lineLength) < FLOAT_EPS)
    {
        return false;
    }

    float u = dot(x, l) / lineLength;

    if (u < 0.0)
    {
        // x is already correct
    }
    else if (u > 1.0)
    {
        x = m - q;
    }
    else // 0.0 < u < 1.0
    {
        x -= u * l;
    }

    return dot(x, x) <= (r * r);
}
