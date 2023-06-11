bool isOutside(vec3 pos, float rad) {
    float dis = dot(pos, camDir);
    if (dis < near - rad || dis > far + rad)
        return true;

    float dis_y = dot(pos, camUp);
    float d_y = rad * frustum_ratio_y;
    float h = dis * frustum_ratio_h;
    if (dis_y > h + d_y || dis_y < -h - d_y)
        return true;

    float dis_x = dot(pos, camRight);
    float d_x = rad * frustum_ratio_x;
    float w = dis * frustum_ratio_w;
    if (dis_x > w + d_x || dis_x < -w - d_x)
        return true;

    return false;

    //return isOutsideZ(pos, rad) || isOutsideY(pos, rad) || isOutsideX(pos, rad);
}

bool isOutsideP(vec4 v0, vec4 v1, vec4 v2, vec4 v3, float rad) {
    if (abs(v0.x) > v0.w || abs(v0.y) > v0.w || abs(v0.z) > v0.w)
        return true;
    if (abs(v1.x) > v1.w || abs(v1.y) > v1.w || abs(v1.z) > v1.w)
        return true;
    if (abs(v2.x) > v2.w || abs(v2.y) > v2.w || abs(v2.z) > v2.w)
        return true;
    if (abs(v3.x) > v3.w || abs(v3.y) > v3.w || abs(v3.z) > v3.w)
        return true;
    return false;
}

bool isOutsideP(vec4 v0, float rad) {
    if (abs(v0.x) > v0.w || abs(v0.y) > v0.w || abs(v0.z) > v0.w)
        return true;
    return false;
}
