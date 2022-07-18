/* main routine */
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // Get pixel coordinates
    vec3 gID = gl_GlobalInvocationID.xyz;
    if (gID.x >= rt_resolution.x || gID.y >= rt_resolution.y) return;

    ivec2 pixel_coords = ivec2(gID.xy);

    // Generate ray
    Ray ray = generateRay(pixel_coords);
    float rayStep = voxelSize * rayStepRatio;
    float rayOffset = wang_hash(pixel_coords.x + pixel_coords.y * uint(rt_resolution.x)) * rayStep;

    // Require tnear or tfar to be positive, so that we can renderer from inside the box,
    // but do not render if the box is completely behind the camera.
    float tnear, tfar;

    if (intersectBox(ray, boxMin, boxMax, tnear, tfar) && (tnear > 0.0f || tfar > 0.0f)) {
        // Initialize ray start and randomly offset it to prevent ringing artifacts
        float t = tnear >= 0.0f ? tnear : 0.0f;
        t += rayOffset;

        // Start computation by calling the function from the specialized shader
        compute(t, tfar, ray, rayStep, pixel_coords);
    } else {
        // Store default values by calling the function from the specialized shader
        storeDefaults(pixel_coords);
    }
}
