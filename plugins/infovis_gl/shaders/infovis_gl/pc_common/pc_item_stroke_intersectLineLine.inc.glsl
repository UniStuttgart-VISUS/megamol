// @see http://stackoverflow.com/a/565282/791895

#define FLOAT_EPS (1.0e-10)

float cross2(vec2 v, vec2 w)
{
    return v.x * w.y - v.y * w.x;
}

bool intersectLineLine(vec2 p, vec2 r, vec2 q, vec2 s)
{
    float rXs = cross2(r, s);

    if (abs(rXs) > FLOAT_EPS)
    {
        vec2 qp = q - p;
        float t = cross2(qp, s) / rXs;
        float u = cross2(qp, r) / rXs;

        return (0.0 <= t) && (t <= 1.0) && (0.0 <= u) && (u <= 1.0);
    }

    return false;
}
