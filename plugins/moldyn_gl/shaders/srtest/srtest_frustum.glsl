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
