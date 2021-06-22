#extension GL_ARB_compute_shader: enable
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

/* view and projection matrices */
uniform mat4 view_mx;
uniform mat4 proj_mx;

/* render target resolution*/
uniform vec2 rt_resolution;

/* bounding box */
uniform vec3 boxMin;
uniform vec3 boxMax;

/* value range */
uniform vec2 valRange;

/* slice */
uniform vec4 slice;

/*	texture that houses the volume data */
uniform highp sampler3D volume_tx3D;

layout(rgba32f, binding = 0) writeonly uniform highp image2D render_target_tx2D;

struct Ray {
    vec3 o;
    vec3 d;
};

bool intersectBox(Ray r, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar) {
    vec3 invR = vec3(1.0f) / r.d;
    vec3 tbot = invR * (boxmin - r.o);
    vec3 ttop = invR * (boxmax - r.o);

    // Special case for a ray lying in a bounding plane.
    if (r.d.x == 0.0f && r.o.x == boxmax.x) {
        ttop.x = -FLT_MAX;
        tbot.x = FLT_MAX;
    }
    if (r.d.y == 0.0f && r.o.y == boxmax.y) {
        ttop.y = -FLT_MAX;
        tbot.y = FLT_MAX;
    }
    if (r.d.z == 0.0f && r.o.z == boxmax.z) {
        ttop.z = -FLT_MAX;
        tbot.z = FLT_MAX;
    }

    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);

    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    tnear = largest_tmin;
    tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    vec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= rt_resolution.x || gID.y >= rt_resolution.y) return;

    ivec2 pixel_coords = ivec2(gID.xy);

    vec2 clip_space_pixel_coords =
        vec2((gID.x / rt_resolution.x) * 2.0f - 1.0f, (gID.y / rt_resolution.y) * 2.0f - 1.0f);

    Ray ray;
    // Unproject a point on the near plane and use as an origin.
    mat4 inv_view_proj_mx = inverse(proj_mx * view_mx);
    vec4 unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, -1.0f, 1.0f);
    ray.o = unproj.xyz / unproj.w;
    // Unproject a point at the same pixel, but further away from the near plane
    // to compute a ray direction in world space.
    unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, 0.0f, 1.0f);
    ray.d = normalize((unproj.xyz / unproj.w) - ray.o);
	
    float tnear, tfar;

    // Require tnear or tfar to be positive, so that we can renderer from inside the box,
    // but do not render if the box is completely behind the camera.
    if (intersectBox(ray, boxMin, boxMax, tnear, tfar) && (tnear > 0.0f || tfar > 0.0f)) {
		ray.o += tnear * ray.d;
		ray.d *= tfar - tnear;

        // Intersect ray with plane
		const vec3 normal = normalize(slice.xyz);

		if (abs(dot(normal, ray.d)) < 0.00001) {
			imageStore(render_target_tx2D, pixel_coords, vec4(0.0f));
			return;
		}

		const float t = (slice.w - dot(normal, ray.o)) / dot(normal, ray.d);

		// Check intersection within the bounding box
		if (t < 0.0f || t > 1.0f) {
			imageStore(render_target_tx2D, pixel_coords, vec4(0.0f));
			return;
		}

		const vec3 contact = ray.o + t * ray.d;

		// Get color
		vec4 result = tflookup((texture(volume_tx3D, (contact - boxMin) / (boxMax - boxMin)).x - valRange.x) / (valRange.y - valRange.x));

		// Store result
        imageStore(render_target_tx2D, pixel_coords, result);
    } else {
        // Always write out to make sure that data from the previous frame is overwritten.
        imageStore(render_target_tx2D, pixel_coords, vec4(0.0f));
    }
}