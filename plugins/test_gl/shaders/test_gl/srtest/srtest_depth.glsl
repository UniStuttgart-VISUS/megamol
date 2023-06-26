float depth(float t) {
    float z_n = (p3_z / t) - p2_z;

    return 0.5f * (z_n + 1.0f);
}
