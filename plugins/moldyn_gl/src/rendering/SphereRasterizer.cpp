#include "SphereRasterizer.h"

#include <chrono>
#include <tuple>

#include <omp.h>
#include <oneapi/tbb.h>

#include <immintrin.h>


float touchplane_old(megamol::moldyn_gl::rendering::SphereRasterizer::config_t const& config, glm::vec3 objPos,
    float sqrRad, float rad, glm::vec3 co_pos) {
#if 0
    // Sphere-Touch-Plane-Approach

    //glm::vec2 winHalf = glm::vec2(config.res.x, config.res.y); // window size

    glm::vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    glm::vec4 tmp;

    glm::vec2 mins, maxs;
    glm::vec3 testPos;
    glm::vec4 projPos;
    float l;

    //float sqrRad = rad * rad;

    // projected camera vector
    glm::vec3 c2 =
        glm::vec3(glm::dot(co_pos, config.camRight), glm::dot(co_pos, config.camUp), glm::dot(co_pos, config.camDir));

    glm::vec3 cpj1 = config.camDir * c2.z + config.camRight * c2.x;
    glm::vec3 cpm1 = config.camDir * c2.x - config.camRight * c2.z;

    glm::vec3 cpj2 = config.camDir * c2.z + config.camUp * c2.y;
    glm::vec3 cpm2 = config.camDir * c2.y - config.camUp * c2.z;

    d.x = glm::length(cpj1);
    d.y = glm::length(cpj2);

    dd = glm::vec2(1.0) / d;

    p = sqrRad * dd;
    q = d - p;
    h = glm::sqrt(p * q);

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    // TODO: rewrite only using four projections, additions in homogenous coordinates and delayed perspective divisions.
    testPos = glm::vec3(objPos) + cpj1 + cpm1;
    projPos = config.MVP * glm::vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = glm::vec2(projPos);
    maxs = glm::vec2(projPos);

    testPos -= glm::vec3(2.0) * cpm1;
    projPos = config.MVP * glm::vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = glm::min(mins, glm::vec2(projPos));
    maxs = glm::max(maxs, glm::vec2(projPos));

    testPos = glm::vec3(objPos) + cpj2 + cpm2;
    projPos = config.MVP * glm::vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = glm::min(mins, glm::vec2(projPos));
    maxs = glm::max(maxs, glm::vec2(projPos));

    testPos -= glm::vec3(2.0) * cpm2;
    projPos = config.MVP * glm::vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = glm::min(mins, glm::vec2(projPos));
    maxs = glm::max(maxs, glm::vec2(projPos));

    /*testPos = glm::vec3(objPos) - config.camDir * rad;
    projPos = config.MVP * glm::vec4(testPos, 1.0);
    projPos /= projPos.w;

    projPos = glm::vec4((mins + maxs) * glm::vec2(0.5), projPos.z, 1.0);*/
    maxs = (maxs - mins) * glm::vec2(0.5) * config.fres;
    l = glm::max(maxs.x, maxs.y) + 0.5;
#else
    glm::vec2 mins, maxs;

    glm::vec3 di = objPos - config.camPos;
    float dd = glm::dot(di, di);

    float s = (sqrRad) / (dd);

    float p = sqrRad / sqrt(dd);

    float v = rad / sqrt(1.0f - s);
    //v = v / sqrt(dd) * (sqrt(dd) - p);
    v = v - v * sqrRad / dd;

    glm::vec3 vr = glm::normalize(glm::cross(di, config.camUp)) * v;
    glm::vec3 vu = glm::normalize(glm::cross(di, vr)) * v;

    glm::vec3 base = objPos - p * config.camDir;

    glm::vec4 v1 = config.MVP * glm::vec4(base + vr, 1.0f);
    glm::vec4 v2 = config.MVP * glm::vec4(base - vr, 1.0f);
    glm::vec4 v3 = config.MVP * glm::vec4(base + vu, 1.0f);
    glm::vec4 v4 = config.MVP * glm::vec4(base - vu, 1.0f);

    v1 /= v1.w;
    v2 /= v2.w;
    v3 /= v3.w;
    v4 /= v4.w;

    mins = glm::vec2(v1);
    maxs = glm::vec2(v1);
    mins = glm::min(mins, glm::vec2(v2));
    maxs = glm::max(maxs, glm::vec2(v2));
    mins = glm::min(mins, glm::vec2(v3));
    maxs = glm::max(maxs, glm::vec2(v3));
    mins = glm::min(mins, glm::vec2(v4));
    maxs = glm::max(maxs, glm::vec2(v4));

    glm::vec2 factor = 0.5f * glm::vec2(config.res);
    v1 = glm::vec4(factor * (glm::vec2(v1) + 1.0f), 0, 0);
    v2 = glm::vec4(factor * (glm::vec2(v2) + 1.0f), 0, 0);
    v3 = glm::vec4(factor * (glm::vec2(v3) + 1.0f), 0, 0);
    v4 = glm::vec4(factor * (glm::vec2(v4) + 1.0f), 0, 0);

    glm::vec2 vw = glm::vec2(v1 - v2);
    glm::vec2 vh = glm::vec2(v3 - v4);

    /*auto projPos = config.MVP * glm::vec4(objPos - p * config.camDir, 1.0f);
    projPos = projPos / projPos.w;*/

    //projPos.xy = (mins + maxs) * 0.5f;
    auto l = glm::max(length(vw), length(vh));
#endif

    return l;
}


bool intersection_old(megamol::moldyn_gl::rendering::SphereRasterizer::config_t const& config, glm::vec3 ray,
    glm::vec3 oc_pos, float sqrRad, float rad, glm::vec3& normal, float& t) {
    // transform fragment coordinates from window coordinates to view coordinates.
    /*vec4 coord =
        vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);*/

    //glm::vec4 coord = fragCoord * glm::vec4(2.0f / static_cast<float>(config.res.x), 2.0f / static_cast<float>(config.res.y), 2.0, 0.0) + glm::vec4(-1.0, -1.0, -1.0, 1.0);
    //auto coord = fragCoord;

    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = config.MVPinv * coord;
    //coord /= coord.w;

    //ray = glm::normalize(glm::vec3(coord) - config.camPos);

    // calculate the geometry-ray-intersection
    float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
    glm::vec3 temp = b * ray - oc_pos;
    float delta = sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

    if (delta < 0.0f)
        return false;

    float c = dot(oc_pos, oc_pos) - sqrRad;

    float s = b < 0.0f ? -1.0f : 1.0f;
    float q = b + s * sqrt(delta);
    t = std::fmin(c / q, q);

    glm::vec3 sphereintersection = t * ray - oc_pos; // intersection point
    normal = (sphereintersection) / rad;

    return true;
}


bool pre_intersection(megamol::moldyn_gl::rendering::SphereRasterizer::config_t const& config, glm::vec3 ray,
    glm::vec3 oc_pos, float sqrRad) {
    // transform fragment coordinates from window coordinates to view coordinates.
    /*vec4 coord =
        vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);*/

    //glm::vec4 coord = fragCoord * glm::vec4(2.0f / static_cast<float>(config.res.x), 2.0f / static_cast<float>(config.res.y), 2.0, 0.0) + glm::vec4(-1.0, -1.0, -1.0, 1.0);
    //auto coord = fragCoord;

    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = config.MVPinv * coord;
    //coord /= coord.w;

    //auto ray = glm::normalize(glm::vec3(coord) - config.camPos);

    // calculate the geometry-ray-intersection
    float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
    glm::vec3 temp = b * ray - oc_pos;
    float delta = sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

    if (delta < 0.0f)
        return false;

    return true;
}


struct Ray {
    glm::vec3 o;
    glm::vec3 d;
};


Ray generateRay(megamol::moldyn_gl::rendering::SphereRasterizer::config_t const& config, glm::ivec2 pixel_coords) {
    Ray ray;

    // Transform pixel to clip coordinates
    glm::vec2 clip_space_pixel_coords = glm::vec2((pixel_coords.x / static_cast<float>(config.res.x)) * 2.0f - 1.0f,
        (pixel_coords.y / static_cast<float>(config.res.y)) * 2.0f - 1.0f);

    // Unproject a point on the near plane and use as an origin
    glm::mat4 inv_view_proj_mx = config.MVPinv;
    glm::vec4 unproj = inv_view_proj_mx * glm::vec4(clip_space_pixel_coords, -1.0f, 1.0f);

    ray.o = glm::vec3(unproj) / unproj.w;

    // Unproject a point at the same pixel, but further away from the near plane
    // to compute a ray direction in world space
    unproj = inv_view_proj_mx * glm::vec4(clip_space_pixel_coords, 0.0f, 1.0f);

    ray.d = glm::vec3(unproj) / unproj.w;
    ray.d = glm::normalize((glm::vec3(unproj) / unproj.w) - ray.o);

    return ray;
}


glm::vec3 LocalLighting(const glm::vec3 ray, const glm::vec3 normal, const glm::vec3 lightdir, const glm::vec3 color) {

    glm::vec3 lightdirn = normalize(-lightdir); // (negativ light dir for directional lighting)

    glm::vec4 lightparams = glm::vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w

    float nDOTl = dot(normal, lightdirn);

    glm::vec3 specular_color = glm::vec3(0.0, 0.0, 0.0);
#ifdef USE_SPECULAR_COMPONENT
    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightdirn);
    specular_color = LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
#endif // USE_SPECULAR_COMPONENT

    return LIGHT_AMBIENT * color + LIGHT_DIFFUSE * color * glm::max(nDOTl, 0.0f) + specular_color;
}


std::vector<glm::u8vec4> megamol::moldyn_gl::rendering::SphereRasterizer::Compute(
    config_t const& config, data_package_t const& data) {
    auto const num_pixels = config.res.x * config.res.y;
    auto const num_particles = data.positions[0].size() / 4;

    //auto const max_threads = omp_get_max_threads();
    auto const max_threads = oneapi::tbb::this_task_arena::max_concurrency();

    std::vector<std::vector<std::tuple<float, int, glm::vec3 /*oc_pos*/, glm::vec3 /*ray*/>>> depth_buffer(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        depth_buffer[i] = std::vector<std::tuple<float, int, glm::vec3 /*oc_pos*/, glm::vec3 /*ray*/>>(
            num_pixels, std::make_tuple(config.near_far.y, -1, glm::vec3(0.0), glm::vec3(0.0)));
    }

    /*float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();*/

    glm::mat4 win_trans = glm::transpose(glm::mat4(glm::vec4(config.res.x * 0.5f, 0, 0, config.res.x * 0.5f),
        glm::vec4(0, config.res.y * 0.5f, 0, config.res.y * 0.5f),
        glm::vec4(0, 0, (config.near_far.y - config.near_far.x) * 0.5f, (config.near_far.y + config.near_far.x) * 0.5f),
        glm::vec4(0, 0, 0, 1)));
    glm::mat4 win_trans_inv = glm::inverse(win_trans);

    //std::cout << "[SphereRasterizer] Max TBB threads " << oneapi::tbb::this_task_arena::max_concurrency() << std::endl;

    float rad = config.global_radius;
    float sqRad = config.global_radius * config.global_radius;

    auto start = std::chrono::high_resolution_clock::now();

    auto vertex_shader = [&config, &data, &sqRad, &rad, &depth_buffer, &win_trans, &win_trans_inv](int item_id) {
        auto const part = glm::vec3(
            data.positions[0][item_id * 4 + 0], data.positions[0][item_id * 4 + 1], data.positions[0][item_id * 4 + 2]);
        /*glm::vec4 projPos;*/
        auto const oc_pos = part - config.camPos;
        auto const l = touchplane_old(config, part, sqRad, rad, -oc_pos);
        // check w
        /*if (projPos.w < -config.near_far.y || projPos.w > -config.near_far.x)
            continue;*/
        // check frustum
        //auto const pw = projPos.w;
        /*projPos = projPos / projPos.w;
        if (projPos.x < -1.0f || projPos.x > 1.0f)
            continue;
        if (projPos.y < -1.0f || projPos.x > 1.0f)
            continue;*/
        // check depth
        /*if (projPos.z < min_z)
            min_z = projPos.z;
        if (projPos.z > max_z)
            max_z = projPos.z;*/

        /*{
            auto pos = config.MVP* glm::vec4(part, 1.0f);
            pos = pos / pos.w;
            pos = win_trans * pos;
            pos = pos * 1.0f;
        }*/
        auto const tmp_dir = glm::normalize(oc_pos);
        auto pos = config.MVP * glm::vec4(part + tmp_dir * rad, 1.0f);
        auto const pw = pos.w;
        pos = pos / pos.w;
        if (pos.x < -1.0f || pos.x > 1.0f)
            return;
        if (pos.y < -1.0f || pos.y > 1.0f)
            return;
        pos = win_trans * pos;
        //auto pos = win_trans * projPos;

        /*auto const screen_x = static_cast<int>((projPos.x + 1.0f) * 0.5f * config.res.x);
        auto const screen_y = static_cast<int>((projPos.y + 1.0f) * 0.5f * config.res.y);*/
        auto const screen_x = static_cast<int>(pos.x);
        auto const screen_y = static_cast<int>(pos.y);
        auto const i_l = static_cast<int>(std::ceilf(l * 0.5f));
        auto const y_low = glm::clamp<int>(screen_y - i_l, 0, config.res.y - 1);
        auto const y_high = glm::clamp<int>(screen_y + i_l, 0, config.res.y - 1);
        auto const x_low = glm::clamp<int>(screen_x - i_l, 0, config.res.x - 1);
        auto const x_high = glm::clamp<int>(screen_x + i_l, 0, config.res.x - 1);
        for (int y = y_low; y <= y_high; ++y) {
            for (int x = x_low; x <= x_high; ++x) {

                auto ndc = win_trans_inv * glm::vec4(x, y, pos.z, 1.0);
                ndc = ndc * pw;
                ndc = config.MVPinv * ndc;
                auto const ray = glm::normalize(glm::vec3(ndc) - config.camPos);
                if (pre_intersection(config, ray, oc_pos, sqRad)) {
                    auto const idx = x + y * config.res.x;
                    auto& d_val = depth_buffer[oneapi::tbb::this_task_arena::current_thread_index()][idx];
                    if (pos.z >= std::get<0>(d_val))
                        continue;
                    d_val = std::make_tuple(pos.z, item_id, oc_pos, ray);
                }
            }
        }
    };

    oneapi::tbb::parallel_for((int)0, (int)num_particles, vertex_shader);

#if 0
#pragma omp parallel for
    for (int i = 0; i < num_particles; ++i) {
        auto const part = glm::vec3(data.positions[0][i * 4 + 0], data.positions[0][i * 4 + 1], data.positions[0][i * 4 + 2]);
        /*glm::vec4 projPos;*/
        auto const oc_pos = part - config.camPos;
        auto const l = touchplane_old(config, part, sqRad, rad, -oc_pos);
        // check w
        /*if (projPos.w < -config.near_far.y || projPos.w > -config.near_far.x)
            continue;*/
        // check frustum
        //auto const pw = projPos.w;
        /*projPos = projPos / projPos.w;
        if (projPos.x < -1.0f || projPos.x > 1.0f)
            continue;
        if (projPos.y < -1.0f || projPos.x > 1.0f)
            continue;*/
        // check depth
        /*if (projPos.z < min_z)
            min_z = projPos.z;
        if (projPos.z > max_z)
            max_z = projPos.z;*/

        /*{
            auto pos = config.MVP* glm::vec4(part, 1.0f);
            pos = pos / pos.w;
            pos = win_trans * pos;
            pos = pos * 1.0f;
        }*/
        auto const tmp_dir = glm::normalize(oc_pos);
        auto pos = config.MVP * glm::vec4(part + tmp_dir * rad, 1.0f);
        auto const pw = pos.w;
        pos = pos / pos.w;
        if (pos.x < -1.0f || pos.x > 1.0f)
            continue;
        if (pos.y < -1.0f || pos.y > 1.0f)
            continue;
        pos = win_trans * pos;
        //auto pos = win_trans * projPos;

        /*auto const screen_x = static_cast<int>((projPos.x + 1.0f) * 0.5f * config.res.x);
        auto const screen_y = static_cast<int>((projPos.y + 1.0f) * 0.5f * config.res.y);*/
        auto const screen_x = static_cast<int>(pos.x);
        auto const screen_y = static_cast<int>(pos.y);
        auto const i_l = static_cast<int>(std::ceilf(l * 0.5f));
        auto const y_low = glm::clamp<int>(screen_y - i_l, 0, config.res.y - 1);
        auto const y_high = glm::clamp<int>(screen_y + i_l, 0, config.res.y - 1);
        auto const x_low = glm::clamp<int>(screen_x - i_l, 0, config.res.x - 1);
        auto const x_high = glm::clamp<int>(screen_x + i_l, 0, config.res.x - 1);
        for (int y = y_low; y <= y_high; ++y) {
            for (int x = x_low; x <= x_high; ++x) {
                
                auto ndc = win_trans_inv * glm::vec4(x, y, pos.z, 1.0);
                ndc = ndc * pw;
                ndc = config.MVPinv * ndc;
                auto const ray = glm::normalize(glm::vec3(ndc) - config.camPos);
                if (pre_intersection(config, ray, oc_pos, sqRad)) {
                    auto const idx = x + y * config.res.x;
                    auto& d_val = depth_buffer[omp_get_thread_num()][idx];
                    if (pos.z >= std::get<0>(d_val))
                        continue;
                    d_val = std::make_tuple(pos.z, i, oc_pos, ray);
                }
            }
        }
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "[SphereRasterizer] vertex processing time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    /*std::cout << "[SphereRasterizer] min_z " << min_z << " max_z " << max_z << " near " << config.near_far.x << " far "
              << config.near_far.y << std::endl;*/

    // reduce buffer
    for (unsigned int y = 0; y < config.res.y; ++y) {
        for (unsigned int x = 0; x < config.res.x; ++x) {
            auto const idx = x + y * config.res.x;
            auto& d_val_0 = depth_buffer[0][idx];
            for (int i = 1; i < max_threads; ++i) {
                auto const& d_val = depth_buffer[i][idx];
                if (std::get<0>(d_val) < std::get<0>(d_val_0))
                    d_val_0 = d_val;
            }
        }
    }

    std::vector<glm::u8vec4> image(num_pixels, glm::u8vec4(0));
    for (unsigned int y = 0; y < config.res.y; ++y) {
        for (unsigned int x = 0; x < config.res.x; ++x) {
            auto const idx = x + y * config.res.x;
            auto const& d_val = depth_buffer[0][idx];
            if (std::get<1>(d_val) == -1)
                continue;
            auto const i = std::get<1>(d_val);
            /*auto const pos =
                glm::vec3(data.positions[0][i * 4 + 0], data.positions[0][i * 4 + 1], data.positions[0][i * 4 + 2]);*/
            glm::vec3 normal;
            float t;

            /*float d = 1.0f / (std::tanf(config.fovy * 0.5f));
            float p_x = x + 0.5f;
            float p_y = y + 0.5f;

            glm::vec3 dir = glm::vec3(config.ratio * (2.0f * p_x / static_cast<float>(config.res.x)) - 1.0f,
                2.0f * p_y / static_cast<float>(config.res.y) - 1.0f, -d);
            dir = glm::normalize(dir);*/

            /*auto r = generateRay(config, glm::ivec2(x, y));

            auto ndc = win_trans_inv * glm::vec4(x, y, std::get<0>(d_val), 1.0);
            ndc = ndc * std::get<3>(d_val);*/

            if (intersection_old(config, std::get<3>(d_val), std::get<2>(d_val), sqRad, rad, normal, t)) {
                auto col = LocalLighting(std::get<3>(d_val), normal, config.lightDir, glm::vec3(1));
                col = col * 255.f;
                image[idx] = glm::u8vec4(
                    static_cast<uint8_t>(col.x), static_cast<uint8_t>(col.y), static_cast<uint8_t>(col.z), 255);
                //image[idx] = glm::u8vec4(255);
            } /*else {
                image[idx] = glm::u8vec4(255, 0, 0, 255);
            }*/
            //image[idx] = glm::u8vec4(255);
            /*r.d = r.d * 255.0f;
            image[idx] = glm::u8vec4(r.d.x, r.d.y, r.d.z, 255);*/
            /*ray = ray * 255.0f;
            image[idx] = glm::u8vec4(ray.x, ray.y, 0, 255);*/
        }
    }
    /*for (int idx = 0; idx < num_pixels; ++idx) {
        auto const& d_val = depth_buffer[idx];
        if (std::get<1>(d_val) == -1)
            continue;
        image[idx] = glm::u8vec4(255);
    }*/

    // per pixel ray casting
    return image;
}
