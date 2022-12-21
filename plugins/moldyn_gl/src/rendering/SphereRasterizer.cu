#include "SphereRasterizer.h"

#include <chrono>
#include <map>
#include <tuple>

#include <omp.h>
#include <oneapi/tbb.h>

#include <immintrin.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/async/scan.h>
#include <thrust/async/sort.h>
#include <thrust/async/transform.h>

#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/mr/pool.h>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>


#define SR_SUBDIVISIONS 640
#define VECTOR_T cuda::vector
#define POLICY thrust::cuda::par(alloc)

struct my_policy : thrust::cuda::execution_policy<my_policy> {};


struct not_my_pointer {
    not_my_pointer(void* p) : message() {
        std::stringstream s;
        s << "Pointer `" << p << "` was not allocated by this allocator.";
        message = s.str();
    }

    virtual ~not_my_pointer() {}

    virtual const char* what() const {
        return message.c_str();
    }

private:
    std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator {
    typedef char value_type;

    cached_allocator() {}

    ~cached_allocator() {
        free_all();
    }

    char* allocate(std::ptrdiff_t num_bytes) {
        std::cout << "cached_allocator::allocate(): num_bytes == " << num_bytes << std::endl;

        char* result = 0;

        // Search the cache for a free block.
        free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

        if (free_block != free_blocks.end()) {
            std::cout << "cached_allocator::allocate(): found a free block" << std::endl;

            result = free_block->second;

            // Erase from the `free_blocks` map.
            free_blocks.erase(free_block);
        } else {
            // No allocation of the right size exists, so create a new one with
            // `thrust::cuda::malloc`.
            try {
                std::cout << "cached_allocator::allocate(): allocating new block" << std::endl;

                // Allocate memory and convert the resulting `thrust::cuda::pointer` to
                // a raw pointer.
                result = thrust::cuda::malloc<char>(num_bytes).get();
            } catch (std::runtime_error&) {
                throw;
            }
        }

        // Insert the allocated pointer into the `allocated_blocks` map.
        allocated_blocks.insert(std::make_pair(result, num_bytes));

        return result;
    }

    void deallocate(char* ptr, size_t) {
        std::cout << "cached_allocator::deallocate(): ptr == " << reinterpret_cast<void*>(ptr) << std::endl;

        // Erase the allocated block from the allocated blocks map.
        allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);

        if (iter == allocated_blocks.end())
            throw not_my_pointer(reinterpret_cast<void*>(ptr));

        std::ptrdiff_t num_bytes = iter->second;
        allocated_blocks.erase(iter);

        // Insert the block into the free blocks map.
        free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all() {
        std::cout << "cached_allocator::free_all()" << std::endl;

        // Deallocate all outstanding blocks in both lists.
        for (free_blocks_type::iterator i = free_blocks.begin(); i != free_blocks.end(); ++i) {
            // Transform the pointer to cuda::pointer before calling cuda::free.
            thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        }

        for (allocated_blocks_type::iterator i = allocated_blocks.begin(); i != allocated_blocks.end(); ++i) {
            // Transform the pointer to cuda::pointer before calling cuda::free.
            thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
        }
    }
};

static cached_allocator alloc;


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


struct worklet {
    glm::uvec4 rectangle;
};


glm::vec4 obj2win(glm::vec4 obj, glm::mat4 mvp, glm::mat4 win_trans) {
    obj = mvp * obj;
    obj /= obj.w;
    return win_trans * obj;
}


std::vector<worklet> create_worklets(megamol::moldyn_gl::rendering::SphereRasterizer::config_t config,
    glm::mat4 win_trans, glm::vec2& final_mins, glm::vec2& final_res, glm::vec2& subs) {
    unsigned int subdivision = SR_SUBDIVISIONS;

    std::vector<worklet> worklets(subdivision * subdivision);

    glm::vec4 v0 = glm::vec4(config.lower, 1);
    glm::vec4 v1 = glm::vec4(glm::vec3(config.lower.x, config.lower.y, config.upper.z), 1);
    glm::vec4 v2 = glm::vec4(glm::vec3(config.lower.x, config.upper.y, config.lower.z), 1);
    glm::vec4 v3 = glm::vec4(glm::vec3(config.upper.x, config.lower.y, config.lower.z), 1);

    glm::vec4 v4 = glm::vec4(glm::vec3(config.upper.x, config.upper.y, config.lower.z), 1);
    glm::vec4 v5 = glm::vec4(glm::vec3(config.upper.x, config.lower.y, config.upper.z), 1);
    glm::vec4 v6 = glm::vec4(glm::vec3(config.lower.x, config.upper.y, config.upper.z), 1);
    glm::vec4 v7 = glm::vec4(config.upper, 1);


    v0 = obj2win(v0, config.MVP, win_trans);
    v1 = obj2win(v1, config.MVP, win_trans);
    v2 = obj2win(v2, config.MVP, win_trans);
    v3 = obj2win(v3, config.MVP, win_trans);
    v4 = obj2win(v4, config.MVP, win_trans);
    v5 = obj2win(v5, config.MVP, win_trans);
    v6 = obj2win(v6, config.MVP, win_trans);
    v7 = obj2win(v7, config.MVP, win_trans);

    glm::vec2 mins = v0;
    glm::vec2 maxs = v0;
    mins = glm::min(mins, glm::vec2(v1));
    maxs = glm::max(maxs, glm::vec2(v1));
    mins = glm::min(mins, glm::vec2(v2));
    maxs = glm::max(maxs, glm::vec2(v2));
    mins = glm::min(mins, glm::vec2(v3));
    maxs = glm::max(maxs, glm::vec2(v3));
    mins = glm::min(mins, glm::vec2(v4));
    maxs = glm::max(maxs, glm::vec2(v4));
    mins = glm::min(mins, glm::vec2(v5));
    maxs = glm::max(maxs, glm::vec2(v5));
    mins = glm::min(mins, glm::vec2(v6));
    maxs = glm::max(maxs, glm::vec2(v6));
    mins = glm::min(mins, glm::vec2(v7));
    maxs = glm::max(maxs, glm::vec2(v7));

    mins = glm::clamp(mins, glm::vec2(0), glm ::vec2(config.res));
    maxs = glm::clamp(maxs, glm::vec2(0), glm ::vec2(config.res));

    glm::vec2 res = maxs - mins;

    auto const sub_x = res.x / subdivision;
    auto const sub_y = res.y / subdivision;

    /*mins = glm::vec2(0);
    maxs = glm::vec2(config.res);
    auto const sub_x = config.res.x / subdivision;
    auto const sub_y = config.res.y / subdivision;*/


    //std::vector<worklet> worklets;
    for (unsigned int y = 0; y < subdivision; ++y) {
        for (unsigned int x = 0; x < subdivision; ++x) {
            glm::uvec4 rec;
            rec.x = mins.x + x * sub_x;
            rec.y = mins.y + y * sub_y;
            if (x < subdivision - 1) {
                rec.z = rec.x + sub_x;
            } else {
                rec.z = maxs.x;
            }
            if (y < subdivision - 1) {
                rec.w = rec.y + sub_y;
            } else {
                rec.w = maxs.y;
            }
            worklet w;
            w.rectangle = rec;
            auto const idx = x + y * subdivision;
            worklets[idx] = w;
            //worklets.emplace_back(w);
        }
    }

    final_mins = mins;
    final_res = res;
    subs = glm::vec2(sub_x, sub_y);

    /*final_mins = glm::vec2(0);
    final_res = glm::vec2(config.res);
    subs = glm::vec2(sub_x, sub_y);*/

    return worklets;
}


struct worker_hash {
    std::vector<float> const& data;
    //std::vector<std::atomic<unsigned int>>& counter;
    std::vector<std::vector<unsigned int>>& counter;

    glm::mat4 win_trans;
    megamol::moldyn_gl::rendering::SphereRasterizer::config_t config;

    float sub_x;
    float sub_y;

    worker_hash(std::vector<float> const& d, std::vector<std::vector<unsigned int>>& c,
        megamol::moldyn_gl::rendering::SphereRasterizer::config_t con, glm::mat4 wt, float sx, float sy)
            : data(d)
            , counter(c)
            , config(con)
            , win_trans(wt)
            , sub_x(sx)
            , sub_y(sy) {}

    void operator()(unsigned int id) {
        auto const part = glm::vec4(data[id * 4 + 0], data[id * 4 + 1], data[id * 4 + 2], 1);
        auto pos = config.MVP * part;
        pos = pos / pos.w;
        /*if (pos.x < -1.0f || pos.x > 1.0f)
            return;
        if (pos.y < -1.0f || pos.y > 1.0f)
            return;*/
        pos = win_trans * pos;

        auto const screen_x = static_cast<int>(pos.x);
        auto const screen_y = static_cast<int>(pos.y);

        auto x = screen_x / sub_x;
        auto y = screen_y / sub_y;

        auto const idx = x + y * SR_SUBDIVISIONS;

        ++counter[oneapi::tbb::this_task_arena::current_thread_index()][idx];
    }
};


struct worker_vertex {
    //worklet worklet_;
    std::vector<float> const& data;
    std::vector<std::tuple<float, int, glm::vec3 /*oc_pos*/, glm::vec3 /*ray*/>>& depth_buffer;

    glm::uvec4 rectangle;
    unsigned int num_particles;
    glm::mat4 win_trans;
    glm::mat4 win_trans_inv;
    float sqRad;
    float rad;
    megamol::moldyn_gl::rendering::SphereRasterizer::config_t config;

    worker_vertex(std::vector<float> const& d,
        std::vector<std::tuple<float, int, glm::vec3 /*oc_pos*/, glm::vec3 /*ray*/>>& db, unsigned int np, float r,
        glm::uvec4 rec, glm::mat4 wt, glm::mat4 wti, megamol::moldyn_gl::rendering::SphereRasterizer::config_t c)
            : data(d)
            , depth_buffer(db)
            , num_particles(np)
            , rad(r)
            , sqRad(r * r)
            , rectangle(rec)
            , win_trans(wt)
            , win_trans_inv(wti)
            , config(c) {}

    void operator()(unsigned int id) {
        /*auto const x = (id % rectangle.z) + rectangle.x;
        auto const y = (id / rectangle.z) + rectangle.y;
        auto const idx = x + y * config.res.x;*/

        /*for (unsigned int i = 0; i < num_particles; ++i)*/ {
            auto const part = glm::vec3(data[id * 4 + 0], data[id * 4 + 1], data[id * 4 + 2]);
            auto const oc_pos = part - config.camPos;
            auto const l = touchplane_old(config, part, sqRad, rad, -oc_pos);

            auto const tmp_dir = glm::normalize(oc_pos);
            auto pos = config.MVP * glm::vec4(part + tmp_dir * rad, 1.0f);
            auto const pw = pos.w;
            pos = pos / pos.w;
            if (pos.x < -1.0f || pos.x > 1.0f)
                return;
            if (pos.y < -1.0f || pos.y > 1.0f)
                return;
            pos = win_trans * pos;

            auto const screen_x = static_cast<int>(pos.x);
            auto const screen_y = static_cast<int>(pos.y);
            auto const i_l = static_cast<int>(std::ceilf(l * 0.5f));
            auto y_low = glm::clamp<int>(screen_y - i_l, 0, config.res.y - 1);
            auto y_high = glm::clamp<int>(screen_y + i_l, 0, config.res.y - 1);
            auto x_low = glm::clamp<int>(screen_x - i_l, 0, config.res.x - 1);
            auto x_high = glm::clamp<int>(screen_x + i_l, 0, config.res.x - 1);

            if (x_low >= rectangle.x && x_low < rectangle.z && y_low >= rectangle.y && y_high < rectangle.w) {
                y_low = glm::clamp<int>(y_low, rectangle.y, rectangle.w - 1);
                y_high = glm::clamp<int>(y_high, rectangle.y, rectangle.w - 1);
                x_low = glm::clamp<int>(x_low, rectangle.x, rectangle.z - 1);
                x_high = glm::clamp<int>(x_high, rectangle.x, rectangle.z - 1);

                for (int y = y_low; y <= y_high; ++y) {
                    for (int x = x_low; x <= x_high; ++x) {
                        auto ndc = win_trans_inv * glm::vec4(x, y, pos.z, 1.0);
                        ndc = ndc * pw;
                        ndc = config.MVPinv * ndc;
                        auto const ray = glm::normalize(glm::vec3(ndc) - config.camPos);
                        if (pre_intersection(config, ray, oc_pos, sqRad)) {
                            auto const idx = x + y * config.res.x;
                            auto& d_val = depth_buffer[idx];
                            if (pos.z < std::get<0>(d_val))
                                d_val = std::make_tuple(pos.z, id, oc_pos, ray);
                        }
                    }
                }
            }
        }
    }
};


struct hash_functor : public thrust::unary_function<glm::vec3, int> {
    glm::mat4 MVP;
    glm::mat4 win_trans;
    glm::vec2 mins;
    glm::vec2 res;
    glm::vec2 subs;
    hash_functor(glm::mat4 MVP, glm::mat4 win_trans, glm::vec2 mins, glm::vec2 res, glm::vec2 subs)
            : MVP(MVP)
            , win_trans(win_trans)
            , mins(mins)
            , res(res)
            , subs(subs) {}
    __host__ __device__ int operator()(glm::vec3 p) const {
        auto pos = MVP * glm::vec4(p, 1);
        pos = pos / pos.w;
        if (pos.x < -1.0f || pos.x > 1.0f)
            return -1;
        if (pos.y < -1.0f || pos.y > 1.0f)
            return -1;
        pos = win_trans * pos;

        auto const screen_x = static_cast<int>(pos.x);
        auto const screen_y = static_cast<int>(pos.y);

        int const x = (screen_x - mins.x) / subs.x;
        int const y = (screen_y - mins.y) / subs.y;

        auto const idx = x + y * SR_SUBDIVISIONS;

        return idx;
    }
};

struct hash_functor2 {
    glm::mat4 MVP;
    glm::mat4 win_trans;
    glm::vec2 mins;
    glm::vec2 res;
    glm::vec2 subs;
    hash_functor2(glm::mat4 MVP, glm::mat4 win_trans, glm::vec2 mins, glm::vec2 res, glm::vec2 subs)
            : MVP(MVP)
            , win_trans(win_trans)
            , mins(mins)
            , res(res)
            , subs(subs) {}
    __host__ __device__ thrust::tuple<int, int> operator()(glm::vec3 const& p, int id) const {
        auto pos = MVP * glm::vec4(p, 1);
        pos = pos / pos.w;
        if (pos.x < -1.0f || pos.x > 1.0f)
            return -1;
        if (pos.y < -1.0f || pos.y > 1.0f)
            return -1;
        pos = win_trans * pos;

        auto const screen_x = static_cast<int>(pos.x);
        auto const screen_y = static_cast<int>(pos.y);

        int const x = (screen_x - mins.x) / subs.x;
        int const y = (screen_y - mins.y) / subs.y;

        auto const idx = x + y * SR_SUBDIVISIONS;

        return thrust::make_tuple(id, idx);
    }
};


struct hash_transform_functor {
    __host__ __device__ glm::ivec3 operator()(int begin, int end) {
        return glm::ivec3(-1, begin, end);
    }
};


struct hash_id_transform_functor {
    __host__ __device__ glm::ivec3 operator()(glm::ivec3& arr, int id) {
        arr.x = id;
        return arr;
    }
};


struct d_db_entry {
    float z;
    int id;
    glm::vec3 oc_pos;
    glm::vec3 ray;
};


struct render_functor {
    glm::vec3 const* data;
    d_db_entry* depth_buffer;

    glm::mat4 MVP;
    glm::mat4 MVPinv;
    glm::mat4 win_trans;
    glm::mat4 win_trans_inv;
    glm::vec3 camPos;
    glm::vec3 camDir;
    glm::vec3 camUp;
    glm::vec3 camRight;
    glm::uvec2 res;

    glm::uvec4 const* rects;

    float sqRad;
    float rad;


    render_functor(glm::vec3 const* data, d_db_entry* depth_buffer, glm::mat4 MVP, glm::mat4 MVPinv,
        glm::mat4 win_trans, glm::mat4 win_trans_inv, glm::vec3 camPos, glm::vec3 camDir, glm::vec3 camUp,
        glm::vec3 camRight, glm::uvec2 res, glm::uvec4 const* rects, float sqRad, float rad)
            : data(data)
            , depth_buffer(depth_buffer)
            , MVP(MVP)
            , MVPinv(MVPinv)
            , win_trans(win_trans)
            , win_trans_inv(win_trans_inv)
            , camPos(camPos)
            , camDir(camDir)
            , camUp(camUp)
            , camRight(camRight)
            , res(res)
            , rects(rects)
            , sqRad(sqRad)
            , rad(rad) {}


    __host__ __device__ float touchplane_old(glm::vec3 const& objPos, glm::vec3 const& co_pos) {
#if 0
        glm::vec2 mins, maxs;

        glm::vec3 di = objPos - camPos;
        float dd = glm::dot(di, di);

        float s = sqRad / dd;

        float p = sqRad / sqrt(dd);

        float v = rad / sqrt(1.0f - s);
        //v = v / sqrt(dd) * (sqrt(dd) - p);
        v = v - v * sqRad / dd;

        glm::vec3 vr = glm::normalize(glm::cross(di, camUp)) * v;
        glm::vec3 vu = glm::normalize(glm::cross(di, vr)) * v;

        glm::vec3 base = objPos - p * camDir;

        glm::vec4 v1 = MVP * glm::vec4(base + vr, 1.0f);
        glm::vec4 v2 = MVP * glm::vec4(base - vr, 1.0f);
        glm::vec4 v3 = MVP * glm::vec4(base + vu, 1.0f);
        glm::vec4 v4 = MVP * glm::vec4(base - vu, 1.0f);

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

        glm::vec2 factor = 0.5f * glm::vec2(res);
        v1 = glm::vec4(factor * (glm::vec2(v1) + 1.0f), 0, 0);
        v2 = glm::vec4(factor * (glm::vec2(v2) + 1.0f), 0, 0);
        v3 = glm::vec4(factor * (glm::vec2(v3) + 1.0f), 0, 0);
        v4 = glm::vec4(factor * (glm::vec2(v4) + 1.0f), 0, 0);

        glm::vec2 vw = glm::vec2(v1 - v2);
        glm::vec2 vh = glm::vec2(v3 - v4);

        auto l = glm::max(length(vw), length(vh));
#else
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
        glm::vec3 c2 = glm::vec3(glm::dot(co_pos, camRight), glm::dot(co_pos, camUp), glm::dot(co_pos, camDir));

        glm::vec3 cpj1 = camDir * c2.z + camRight * c2.x;
        glm::vec3 cpm1 = camDir * c2.x - camRight * c2.z;

        glm::vec3 cpj2 = camDir * c2.z + camUp * c2.y;
        glm::vec3 cpm2 = camDir * c2.y - camUp * c2.z;

        d.x = glm::length(cpj1);
        d.y = glm::length(cpj2);

        dd = glm::vec2(1.0) / d;

        p = sqRad * dd;
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
        projPos = MVP * glm::vec4(testPos, 1.0);
        projPos /= projPos.w;
        mins = glm::vec2(projPos);
        maxs = glm::vec2(projPos);

        testPos -= glm::vec3(2.0) * cpm1;
        projPos = MVP * glm::vec4(testPos, 1.0);
        projPos /= projPos.w;
        mins = glm::min(mins, glm::vec2(projPos));
        maxs = glm::max(maxs, glm::vec2(projPos));

        testPos = glm::vec3(objPos) + cpj2 + cpm2;
        projPos = MVP * glm::vec4(testPos, 1.0);
        projPos /= projPos.w;
        mins = glm::min(mins, glm::vec2(projPos));
        maxs = glm::max(maxs, glm::vec2(projPos));

        testPos -= glm::vec3(2.0) * cpm2;
        projPos = MVP * glm::vec4(testPos, 1.0);
        projPos /= projPos.w;
        mins = glm::min(mins, glm::vec2(projPos));
        maxs = glm::max(maxs, glm::vec2(projPos));

        /*testPos = glm::vec3(objPos) - config.camDir * rad;
        projPos = config.MVP * glm::vec4(testPos, 1.0);
        projPos /= projPos.w;

        projPos = glm::vec4((mins + maxs) * glm::vec2(0.5), projPos.z, 1.0);*/
        maxs = (maxs - mins) * glm::vec2(0.5) * glm::vec2(res);
        l = glm::max(maxs.x, maxs.y) + 0.5;
#endif

        return l;
    }


    __host__ __device__ bool pre_intersection(glm::vec3 ray, glm::vec3 oc_pos, float sqrRad) {
        // calculate the geometry-ray-intersection
        float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
        glm::vec3 temp = b * ray - oc_pos;
        float delta =
            sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

        if (delta < 0.0f)
            return false;

        return true;
    }


    __host__ __device__ void operator()(glm::ivec3 const& job) {
        auto w_id = job.x;
        auto const p_begin = job.y;
        auto const p_end = job.z;

        if (w_id < 0)
            return;

        /*if ((w_id % SR_SUBDIVISIONS)%2 == 1)
            return;*/

        //glm::uvec4 const rectangle = rects[w_id];

        for (int p_idx = p_begin; p_idx < p_end; ++p_idx) {
            glm::vec3 const p = data[p_idx];

            auto const oc_pos = p - camPos;
            auto const l = touchplane_old(p, -oc_pos);

            auto const tmp_dir = glm::normalize(oc_pos);
            auto pos = MVP * glm::vec4(p + tmp_dir * rad, 1.0f);
            auto const pw = pos.w;
            pos = pos / pos.w;
            pos = win_trans * pos;

            auto const screen_x = static_cast<int>(pos.x);
            auto const screen_y = static_cast<int>(pos.y);
            /*if (screen_x < rectangle.x || screen_x >= rectangle.z || screen_y < rectangle.y ||
                screen_y >= rectangle.w)
                continue;*/
            auto const i_l = static_cast<int>(std::ceilf(l * 0.5f));
            auto y_low = glm::clamp<int>(screen_y - i_l, 0, res.y - 1);
            auto y_high = glm::clamp<int>(screen_y + i_l, 0, res.y - 1);
            auto x_low = glm::clamp<int>(screen_x - i_l, 0, res.x - 1);
            auto x_high = glm::clamp<int>(screen_x + i_l, 0, res.x - 1);

            //if (x_low >= rectangle.x && x_low < rectangle.z && y_low >= rectangle.y && y_high < rectangle.w) {
            /*y_low = glm::clamp<int>(y_low, rectangle.y, rectangle.w - 1);
            y_high = glm::clamp<int>(y_high, rectangle.y, rectangle.w - 1);
            x_low = glm::clamp<int>(x_low, rectangle.x, rectangle.z - 1);
            x_high = glm::clamp<int>(x_high, rectangle.x, rectangle.z - 1);*/

            for (int y = y_low; y <= y_high; ++y) {
                for (int x = x_low; x <= x_high; ++x) {
                    auto ndc = win_trans_inv * glm::vec4(x, y, pos.z, 1.0);
                    ndc = ndc * pw;
                    ndc = MVPinv * ndc;
                    auto const ray = glm::normalize(glm::vec3(ndc) - camPos);
                    if (pre_intersection(ray, oc_pos, sqRad)) {
                        auto const idx = x + y * res.x;
                        d_db_entry& d_val = depth_buffer[idx];
                        if (pos.z < d_val.z) {
                            d_val.z = pos.z;
                            d_val.id = p_idx;
                            d_val.oc_pos = oc_pos;
                            d_val.ray = ray;
                        }
                        //depth_buffer[idx] = d_val;
                    }
                }
            }
            //}
        }
    }
};


struct blit_functor {
    d_db_entry* depth_buffer;

    glm::vec3 lightDir;

    float sqRad;
    float rad;

    blit_functor(d_db_entry* depth_buffer, glm::vec3 lightDir, float sqRad, float rad)
            : depth_buffer(depth_buffer)
            , lightDir(lightDir)
            , sqRad(sqRad)
            , rad(rad) {}

    __host__ __device__ bool intersection_old(
        glm::vec3 ray, glm::vec3 oc_pos, float sqrRad, float rad, glm::vec3& normal, float& t) const {
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
        float delta =
            sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

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

    __host__ __device__ glm::vec3 LocalLighting(
        const glm::vec3 ray, const glm::vec3 normal, const glm::vec3 lightdir, const glm::vec3 color) const {

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

    __host__ __device__ glm::uvec4 operator()(d_db_entry const& d_val) const {
        //auto const idx = x + y * res.x;
        //auto const& d_val = depth_buffer[idx];
        if (d_val.id == -1)
            return glm::uvec4(0, 0, 0, 255);
        auto const i = d_val.id;
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

        if (intersection_old(d_val.ray, d_val.oc_pos, sqRad, rad, normal, t)) {
            auto col = LocalLighting(d_val.ray, normal, lightDir, glm::vec3(1));
            col = col * 255.f;
            return glm::u8vec4(
                static_cast<uint8_t>(col.x), static_cast<uint8_t>(col.y), static_cast<uint8_t>(col.z), 255);
            //image[idx] = glm::u8vec4(255);
        }
        return glm::uvec4(0, 0, 0, 1);
    }
};


struct render_functor2 {
    glm::vec3 const* data;
    d_db_entry* depth_buffer;
    thrust::tuple<int, int>* ids;

    glm::mat4 MVP;
    glm::mat4 MVPinv;
    glm::mat4 win_trans;
    glm::mat4 win_trans_inv;
    glm::vec3 camPos;
    glm::vec3 camDir;
    glm::vec3 camUp;
    glm::uvec2 res;

    //glm::uvec4 const* rects;

    float sqRad;
    float rad;


    render_functor2(glm::vec3 const* data, d_db_entry* depth_buffer, thrust::tuple<int, int>* ids, glm::mat4 MVP,
        glm::mat4 MVPinv, glm::mat4 win_trans, glm::mat4 win_trans_inv, glm::vec3 camPos, glm::vec3 camDir,
        glm::vec3 camUp, glm::uvec2 res, glm::uvec4 const* rects, float sqRad, float rad)
            : data(data)
            , depth_buffer(depth_buffer)
            , ids(ids)
            , MVP(MVP)
            , MVPinv(MVPinv)
            , win_trans(win_trans)
            , win_trans_inv(win_trans_inv)
            , camPos(camPos)
            , camDir(camDir)
            , camUp(camUp)
            , res(res)
            //, rects(rects)
            , sqRad(sqRad)
            , rad(rad) {}


    __host__ __device__ float touchplane_old(glm::vec3 const& objPos) {
        glm::vec2 mins, maxs;

        glm::vec3 di = objPos - camPos;
        float dd = glm::dot(di, di);

        float s = sqRad / dd;

        float p = sqRad / sqrt(dd);

        float v = rad / sqrt(1.0f - s);
        //v = v / sqrt(dd) * (sqrt(dd) - p);
        v = v - v * sqRad / dd;

        glm::vec3 vr = glm::normalize(glm::cross(di, camUp)) * v;
        glm::vec3 vu = glm::normalize(glm::cross(di, vr)) * v;

        glm::vec3 base = objPos - p * camDir;

        glm::vec4 v1 = MVP * glm::vec4(base + vr, 1.0f);
        glm::vec4 v2 = MVP * glm::vec4(base - vr, 1.0f);
        glm::vec4 v3 = MVP * glm::vec4(base + vu, 1.0f);
        glm::vec4 v4 = MVP * glm::vec4(base - vu, 1.0f);

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

        glm::vec2 factor = 0.5f * glm::vec2(res);
        v1 = glm::vec4(factor * (glm::vec2(v1) + 1.0f), 0, 0);
        v2 = glm::vec4(factor * (glm::vec2(v2) + 1.0f), 0, 0);
        v3 = glm::vec4(factor * (glm::vec2(v3) + 1.0f), 0, 0);
        v4 = glm::vec4(factor * (glm::vec2(v4) + 1.0f), 0, 0);

        glm::vec2 vw = glm::vec2(v1 - v2);
        glm::vec2 vh = glm::vec2(v3 - v4);

        auto l = glm::max(length(vw), length(vh));

        return l;
    }


    __host__ __device__ bool pre_intersection(glm::vec3 ray, glm::vec3 oc_pos, float sqrRad) {
        // calculate the geometry-ray-intersection
        float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
        glm::vec3 temp = b * ray - oc_pos;
        float delta =
            sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

        if (delta < 0.0f)
            return false;

        return true;
    }


    __host__ __device__ void operator()(glm::ivec3 const& job) {
        auto w_id = job.x;
        auto const p_begin = job.y;
        auto const p_end = job.z;

        if (w_id < 0)
            return;

        /*if ((w_id % SR_SUBDIVISIONS)%2 == 1)
            return;*/

        //glm::uvec4 const rectangle = rects[w_id];

        for (int p_idx = p_begin; p_idx < p_end; ++p_idx) {
            glm::vec3 const p = data[thrust::get<0>(ids[p_idx])];

            auto const oc_pos = p - camPos;
            auto const l = touchplane_old(p);

            auto const tmp_dir = glm::normalize(oc_pos);
            auto pos = MVP * glm::vec4(p + tmp_dir * rad, 1.0f);
            auto const pw = pos.w;
            pos = pos / pos.w;
            pos = win_trans * pos;

            auto const screen_x = static_cast<int>(pos.x);
            auto const screen_y = static_cast<int>(pos.y);
            /*if (screen_x < rectangle.x || screen_x >= rectangle.z || screen_y < rectangle.y ||
                screen_y >= rectangle.w)
                continue;*/
            auto const i_l = static_cast<int>(std::ceilf(l * 0.5f));
            auto y_low = glm::clamp<int>(screen_y - i_l, 0, res.y - 1);
            auto y_high = glm::clamp<int>(screen_y + i_l, 0, res.y - 1);
            auto x_low = glm::clamp<int>(screen_x - i_l, 0, res.x - 1);
            auto x_high = glm::clamp<int>(screen_x + i_l, 0, res.x - 1);

            //if (x_low >= rectangle.x && x_low < rectangle.z && y_low >= rectangle.y && y_high < rectangle.w) {
            /*y_low = glm::clamp<int>(y_low, rectangle.y, rectangle.w - 1);
            y_high = glm::clamp<int>(y_high, rectangle.y, rectangle.w - 1);
            x_low = glm::clamp<int>(x_low, rectangle.x, rectangle.z - 1);
            x_high = glm::clamp<int>(x_high, rectangle.x, rectangle.z - 1);*/

            for (int y = y_low; y <= y_high; ++y) {
                for (int x = x_low; x <= x_high; ++x) {
                    auto ndc = win_trans_inv * glm::vec4(x, y, pos.z, 1.0);
                    ndc = ndc * pw;
                    ndc = MVPinv * ndc;
                    auto const ray = glm::normalize(glm::vec3(ndc) - camPos);
                    if (pre_intersection(ray, oc_pos, sqRad)) {
                        auto const idx = x + y * res.x;
                        d_db_entry& d_val = depth_buffer[idx];
                        if (pos.z < d_val.z) {
                            d_val.z = pos.z;
                            d_val.id = p_idx;
                            d_val.oc_pos = oc_pos;
                            d_val.ray = ray;
                        }
                        //depth_buffer[idx] = d_val;
                    }
                }
            }
            //}
        }
    }
};


struct sort_functor {
    __host__ __device__ bool operator()(thrust::tuple<int, int> const& lhs, thrust::tuple<int, int> const& rhs) const {
        return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    }
};


struct copy_functor {
    __host__ __device__ int operator()(thrust::tuple<int, int> const& lhs) const {
        return thrust::get<1>(lhs);
    }
};


struct bound_functor {
    __host__ __device__ bool operator()(int lhs, thrust::tuple<int, int> const& rhs) const {
        return lhs < thrust::get<1>(rhs);
    }
};


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

    glm::vec2 f_mins;
    glm::vec2 f_res;
    glm::vec2 f_subs;

    auto worklets = create_worklets(config, win_trans, f_mins, f_res, f_subs);
    std::vector<glm::uvec4> v_worklets(worklets.size());
    std::transform(worklets.begin(), worklets.end(), v_worklets.begin(), [](worklet const& w) { return w.rectangle; });
    //std::cout << "[SphereRasterizer] Max TBB threads " << oneapi::tbb::this_task_arena::max_concurrency() << std::endl;

    std::cout << "[SphereRasterizer] Res: " << f_res.x << " " << f_res.y << std::endl;
    std::cout << "[SphereRasterizer] Mins: " << f_mins.x << " " << f_mins.y << std::endl;
    std::cout << "[SphereRasterizer] Subs: " << f_subs.x << " " << f_subs.y << std::endl;

    float rad = config.global_radius;
    float sqRad = config.global_radius * config.global_radius;

    std::vector<std::vector<unsigned int>> counter(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        counter[i] = std::vector<unsigned int>(SR_SUBDIVISIONS * SR_SUBDIVISIONS);
    }

    std::vector<glm::vec3> tmp_data(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        tmp_data[i] =
            glm::vec3(data.positions[0][i * 4 + 0], data.positions[0][i * 4 + 1], data.positions[0][i * 4 + 2]);
    }

    using Pool = thrust::mr::disjoint_unsynchronized_pool_resource<thrust::device_memory_resource,
        thrust::mr::new_delete_resource>;
    using IntAlloc = thrust::mr::allocator<int, Pool>;
    using Vec3Alloc = thrust::mr::allocator<glm::vec3, Pool>;
    using IVec3Alloc = thrust::mr::allocator<glm::ivec3, Pool>;
    using UVec4Alloc = thrust::mr::allocator<glm::uvec4, Pool>;
    using DBAlloc = thrust::mr::allocator<d_db_entry, Pool>;

    thrust::device_memory_resource dev_memres;
    thrust::mr::new_delete_resource memres;
    Pool pool(&dev_memres, &memres);
    IntAlloc int_alloc(&pool);
    Vec3Alloc vec3_alloc(&pool);
    IVec3Alloc ivec3_alloc(&pool);
    UVec4Alloc uvec4_alloc(&pool);
    DBAlloc db_alloc(&pool);


    thrust::VECTOR_T<glm::vec3> d_in_data(tmp_data.begin(), tmp_data.end());
    thrust::VECTOR_T<int> d_cell_id(num_particles, -1);

    /*auto const sub_x = config.res.x / SR_SUBDIVISIONS;
    auto const sub_y = config.res.y / SR_SUBDIVISIONS;*/

    auto hash_func = hash_functor(config.MVP, win_trans, f_mins, f_res, f_subs);
    auto hash_func2 = hash_functor2(config.MVP, win_trans, f_mins, f_res, f_subs);

    thrust::VECTOR_T<int> d_histo(SR_SUBDIVISIONS * SR_SUBDIVISIONS + 2, 0);
    thrust::VECTOR_T<int> d_h_begin(SR_SUBDIVISIONS * SR_SUBDIVISIONS + 1);
    thrust::VECTOR_T<int> d_h_end(SR_SUBDIVISIONS * SR_SUBDIVISIONS + 1);
    thrust::VECTOR_T<glm::ivec3> d_h_id_begin_end(SR_SUBDIVISIONS * SR_SUBDIVISIONS);

    d_db_entry fill_db_entry;
    fill_db_entry.z = config.near_far.y;
    fill_db_entry.id = -1;
    fill_db_entry.oc_pos = glm::vec3(0.0);
    fill_db_entry.ray = glm::vec3(0.0);

    thrust::VECTOR_T<d_db_entry> d_depth_buffer(num_pixels, fill_db_entry);
    thrust::VECTOR_T<glm::uvec4> d_worklets(v_worklets.begin(), v_worklets.end());
    thrust::VECTOR_T<glm::uvec4> d_image(num_pixels, glm::uvec4(0));

    thrust::VECTOR_T<int> test_list(num_particles);

    auto render_func =
        render_functor(thrust::raw_pointer_cast(&d_in_data[0]), thrust::raw_pointer_cast(&d_depth_buffer[0]),
            config.MVP, config.MVPinv, win_trans, win_trans_inv, config.camPos, config.camDir, config.camUp,
            config.camRight, config.res, thrust::raw_pointer_cast(&d_worklets[0]), sqRad, rad);

    auto blit_func = blit_functor(thrust::raw_pointer_cast(&d_depth_buffer[0]), config.lightDir, sqRad, rad);

    /*cudaStream_t s1;
    cudaStreamCreate(&s1);*/

    thrust::VECTOR_T<int> p_id(num_particles);
    thrust::VECTOR_T<thrust::tuple<int, int>> t_cell(num_particles);

    auto render_func2 =
        render_functor2(thrust::raw_pointer_cast(&d_in_data[0]), thrust::raw_pointer_cast(&d_depth_buffer[0]),
            thrust::raw_pointer_cast(&t_cell[0]), config.MVP, config.MVPinv, win_trans, win_trans_inv, config.camPos,
            config.camDir, config.camUp, config.res, thrust::raw_pointer_cast(&d_worklets[0]), sqRad, rad);

    thrust::copy(
        thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_particles), p_id.begin());
//thrust::copy(p_id.begin(), p_id.begin()+10, std::ostream_iterator<int>(std::cout, " "));


/*auto s_iter_begin = thrust::make_zip_iterator(thrust::make_tuple(p_id.begin(), thrust::make_transform_iterator(d_in_data.begin(), hash_func)));
auto s_iter_end = thrust::make_zip_iterator(thrust::make_tuple(p_id.end(), thrust::make_transform_iterator(d_in_data.end(), hash_func)));*/
#if 0
    auto start_s = std::chrono::high_resolution_clock::now();
    thrust::transform(
        POLICY, d_in_data.begin(), d_in_data.end(), thrust::counting_iterator<int>(0), t_cell.begin(), hash_func2);
    thrust::sort(POLICY, t_cell.begin(), t_cell.end(), sort_functor());
    thrust::transform(POLICY, t_cell.begin(), t_cell.end(), test_list.begin(), copy_functor());
    test_list.erase(thrust::unique(POLICY, test_list.begin(), test_list.end()), test_list.end());
    thrust::upper_bound(
        POLICY, t_cell.begin(), t_cell.end(), test_list.begin(), test_list.end(), d_histo.begin() + 1, bound_functor());
    thrust::transform(POLICY, d_histo.begin(), d_histo.begin() + test_list.size(), d_histo.begin() + 1,
        d_h_id_begin_end.begin(), hash_transform_functor());
    thrust::transform(POLICY, d_h_id_begin_end.begin(), d_h_id_begin_end.begin() + test_list.size(), test_list.begin(),
        d_h_id_begin_end.begin(), hash_id_transform_functor());
    //cudaDeviceSynchronize();
    thrust::for_each(POLICY, d_h_id_begin_end.begin(), d_h_id_begin_end.begin() + test_list.size(), render_func2);
    cudaDeviceSynchronize();
    auto end_s = std::chrono::high_resolution_clock::now();
    std::cout << "[SphereRasterizer] test processing time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_s - start_s).count() << "ms" << std::endl;
#endif
#if 1


    /*auto temp_ptr = alloc.allocate(1024 * 1024 * 1024);
    alloc.deallocate(temp_ptr, 1024 * 1024 * 1024);*/

    auto start_x = std::chrono::high_resolution_clock::now();
#ifdef MEGAMOL_USE_PROFILING
    pm->set_transient_comment((*timing_handles_)[3], "thrust");
    pm->start_timer((*timing_handles_)[3]);
#endif
    thrust::transform(POLICY, d_in_data.begin(), d_in_data.end(), d_cell_id.begin(), hash_func);
    thrust::sort_by_key(POLICY, d_cell_id.begin(), d_cell_id.end(), d_in_data.begin());
    //auto first_it = thrust::find_if(d_cell_id.begin(), d_cell_id.end(), [](auto const& val) { return val != -1; });
    //auto offset = thrust::distance(d_cell_id.begin(), first_it);
    //auto offset = 0;
    thrust::copy(POLICY, d_cell_id.begin(), d_cell_id.end(), test_list.begin());
    test_list.erase(thrust::unique(POLICY, test_list.begin(), test_list.end()), test_list.end());
    //thrust::counting_iterator<int> search_begin(-1);
    /*thrust::upper_bound(d_cell_id.begin() + offset, d_cell_id.end(), search_begin,
        search_begin + (SR_SUBDIVISIONS * SR_SUBDIVISIONS) + 1, d_histo.begin());*/
    thrust::upper_bound(
        POLICY, d_cell_id.begin(), d_cell_id.end(), test_list.begin(), test_list.end(), d_histo.begin() + 1);
    //d_histo.erase(d_histo.begin() + test_list.size(), d_histo.end());
#if 0
    thrust::adjacent_difference(POLICY, d_histo.begin(), d_histo.begin() + test_list.size(), d_histo.begin());
    //d_h_begin = thrust::VECTOR_T<int>(test_list.size());
    //d_h_end = thrust::VECTOR_T<int>(test_list.size());
    thrust::exclusive_scan(POLICY, d_histo.begin(), d_histo.begin() + test_list.size(), d_h_begin.begin());
    thrust::inclusive_scan(POLICY, d_histo.begin(), d_histo.begin() + test_list.size(), d_h_end.begin());
    //d_h_id_begin_end = thrust::VECTOR_T<std::array<int, 3>>(test_list.size());
    thrust::transform(POLICY, d_h_begin.begin(), d_h_begin.begin() + test_list.size(), d_h_end.begin(),
        d_h_id_begin_end.begin(), hash_transform_functor());
    //thrust::counting_iterator<int> hash_begin(0);
    /*thrust::transform(d_h_id_begin_end.begin(), d_h_id_begin_end.end(), hash_begin,
        d_h_id_begin_end.begin(), hash_id_transform_functor());*/
#endif
    thrust::transform(POLICY, d_histo.begin(), d_histo.begin() + test_list.size(), d_histo.begin() + 1,
        d_h_id_begin_end.begin(), hash_transform_functor());
    thrust::transform(POLICY, d_h_id_begin_end.begin(), d_h_id_begin_end.begin() + test_list.size(), test_list.begin(),
        d_h_id_begin_end.begin(), hash_id_transform_functor());
    //cudaDeviceSynchronize();
    thrust::for_each(POLICY, d_h_id_begin_end.begin(), d_h_id_begin_end.begin() + test_list.size(), render_func);
    thrust::transform(POLICY, d_depth_buffer.begin(), d_depth_buffer.end(), d_image.begin(), blit_func);
    cudaDeviceSynchronize();
#ifdef MEGAMOL_USE_PROFILING
    pm->stop_timer((*timing_handles_)[3]);
#endif
    auto end_x = std::chrono::high_resolution_clock::now();

#if 0
    worker_hash w(data.positions[0], counter, config, win_trans, config.res.x / 20, config.res.y / 20);
    oneapi::tbb::parallel_for((unsigned int)0, (unsigned int)num_particles, [&w](unsigned int i) { w(i); });

    oneapi::tbb::parallel_for((unsigned int)0, (unsigned int)20 * 20, [&counter, &max_threads](unsigned int i) {
        for (int j = 1; j < max_threads; ++j) {
            counter[0][i] += counter[j][i];
        }
    });

    std::vector<std::vector<unsigned int>> id_list(20 * 20);
    std::vector<unsigned int> idx_vec(20 * 20, 0);
    for (int i = 0; i < 20 * 20; ++i) {
        id_list.reserve(counter[0][i]);
    }
    //oneapi::tbb::parallel_for((unsigned int)0, (unsigned int)(20 * 20), [&id_list, &idx_vec](unsigned int i) {
    //    auto const part = glm::vec4(data[id * 4 + 0], data[id * 4 + 1], data[id * 4 + 2], 1);
    //    auto pos = config.MVP * part;
    //    pos = pos / pos.w;
    //    /*if (pos.x < -1.0f || pos.x > 1.0f)
    //        return;
    //    if (pos.y < -1.0f || pos.y > 1.0f)
    //        return;*/
    //    pos = win_trans * pos;

    //    auto const screen_x = static_cast<int>(pos.x);
    //    auto const screen_y = static_cast<int>(pos.y);

    //    auto x = screen_x / sub_x;
    //    auto y = screen_y / sub_y;

    //    auto const idx = x + y * 20;

    //    id_list[idx][idx_vec[idx]] = id;
    //    });
#endif


    std::cout << "[SphereRasterizer] hashing processing time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_x - start_x).count() << "ms" << std::endl;
#endif
    //thrust::copy(d_histo.begin(), d_histo.end(), std::ostream_iterator<int>(std::cout, " "));
    /*std::cout << std::endl;
    for (auto const& el : d_h_id_begin_end) {
        std::cout << el[2] << " ";
    }
    std::cout << std::endl;*/

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

    //oneapi::tbb::parallel_for((int)0, (int)num_particles, vertex_shader);

    /*oneapi::tbb::parallel_for((int)0, (int)worklets.size(),
        [&num_particles, &rad, &worklets, &data, &depth_buffer, &win_trans, &win_trans_inv, &config](int work_id) {
            auto worklet = worklets[work_id];
            worker_vertex w(data.positions[0], depth_buffer[0], num_particles, rad, worklet.rectangle, win_trans,
                win_trans_inv, config);
            auto width = worklet.rectangle.z - worklet.rectangle.x;
            auto height = worklet.rectangle.w - worklet.rectangle.y;
            oneapi::tbb::parallel_for((unsigned int)0, (unsigned int)(num_particles), [&w](unsigned int i) { w(i); });
        });*/

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
    /*for (unsigned int y = 0; y < config.res.y; ++y) {
        for (unsigned int x = 0; x < config.res.x; ++x) {
            auto const idx = x + y * config.res.x;
            auto& d_val_0 = depth_buffer[0][idx];
            for (int i = 1; i < max_threads; ++i) {
                auto const& d_val = depth_buffer[i][idx];
                if (std::get<0>(d_val) < std::get<0>(d_val_0))
                    d_val_0 = d_val;
            }
        }
    }*/
    std::vector<d_db_entry> tmp_depth_buffer(num_pixels);
    thrust::copy(d_depth_buffer.begin(), d_depth_buffer.end(), tmp_depth_buffer.begin());
    std::transform(tmp_depth_buffer.begin(), tmp_depth_buffer.end(), depth_buffer[0].begin(),
        [](d_db_entry const& entry) { return std::make_tuple(entry.z, entry.id, entry.oc_pos, entry.ray); });

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

    thrust::copy(d_image.begin(), d_image.end(), image.begin());

    // per pixel ray casting
    return image;
}
