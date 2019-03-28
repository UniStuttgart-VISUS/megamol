#include "streamlines.h"
#include "streamlines.cuh"

#include <cuda_runtime_api.h>

#include "real_type.h"
#include "functions.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#if (false) // Used to disable errors in IDE. Turn off when compiling!
template <typename t> t tex1Dfetch(cudaTextureObject_t, int);
int2 threadIdx, blockIdx, blockDim;
#define __cuda_kernel_start(a, b)
#else
#define __cuda_kernel_start(a, b) <<< a, b >>>
#endif

// Transformations: domain offset, domain scale, texture offset, texture scale, time scale, max integration error}
__constant__ float2 const_data[6];

// Textures: vector field, 4th-order runge-kutta step size, convergence points, point ids, convergence lines, line ids
__constant__ cudaTextureObject_t textures[6];

/**
* Transform world position to texture coordinates
*
* @param pos World position
*
* @return Texture coordinates
*/
inline __device__
float2 pos_to_texcoords(const float2 pos)
{
    // Transform position from [physical] to [0 : 1]
    const float2 scaled_position = (pos - const_data[0]) * const_data[1];

    // Transform position from [0 : 1] to [0.5 : texture width - 0.5]
    return scaled_position * const_data[3] - const_data[2];
}

/**
* Transform world position to surface memory coordinates
*
* @param pos World position
*
* @return Surface memory coordinates
*/
inline __device__
int2 pos_to_surfcoords(const float2 pos)
{
    // Transform position from [physical] to [0 : 1]
    float2 scaled_position = (pos - const_data[0]) * const_data[1];

    scaled_position = fmaxf(fminf(scaled_position, make_real<float, 2>(1.0f)), make_real<float, 2>(0.0f));

    // Transform position from [0 : 1] to [0 : texWidth - 1]
    return make_int2(floor(scaled_position.x * const_data[3].x), floor(scaled_position.y * const_data[3].y));
}

#if !(__streamlines_cuda_runge_kutta_45)
/**
* Advect position using 4th-order Runge-Kutta
*
* @param pos    In/out position
* @param delta  Time step coefficient
* @param sign   Sign (1: forward, 2: backward integration)
*/
__device__
void advectRK4(float2& pos, const float delta, const float sign)
{
    // Calculate Runge-Kutta coefficients
    float2 velocity, scaled_position;

    const float step = texture_interpolation<1>(textures[1], pos_to_texcoords(pos));

    const float2 k1 = step * delta * normalizeSafe(sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos)));
    const float2 k2 = step * delta * normalizeSafe(sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + 0.5f * k1)));
    const float2 k3 = step * delta * normalizeSafe(sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + 0.5f * k2)));
    const float2 k4 = step * delta * normalizeSafe(sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + k3)));

    // Advect and store position
    pos += (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}
#else
/**
* Advect position using 4th-order Runge-Kutta with 5th-order error estimation for adaptive time steps
* See http://www.aip.de/groups/soe/local/numres/bookcpdf/c16-2.pdf for details
*
* @param pos        In/out position
* @param delta      Time step coefficient
* @param sign       Sign (1: forward, 2: backward integration)
* @param max_error  Maximum error allowed, used for step size adjustment
*
* @return Time step coefficient based on error estimation, with corresponding error
*/
__device__
float2 advectRK45(float2& pos, float& delta, const float sign, const float max_error)
{
    // Cash-Karp parameters
    constexpr float b_21 = 0.2f;
    constexpr float b_31 = 0.075f;
    constexpr float b_41 = 0.3f;
    constexpr float b_51 = -11.0f / 54.0f;
    constexpr float b_61 = 1631.0f / 55296.0f;
    constexpr float b_32 = 0.225f;
    constexpr float b_42 = -0.9f;
    constexpr float b_52 = 2.5f;
    constexpr float b_62 = 175.0f / 512.0f;
    constexpr float b_43 = 1.2f;
    constexpr float b_53 = -70.0f / 27.0f;
    constexpr float b_63 = 575.0f / 13824.0f;
    constexpr float b_54 = 35.0f / 27.0f;
    constexpr float b_64 = 44275.0f / 110592.0f;
    constexpr float b_65 = 253.0f / 4096.0f;

    constexpr float c_1 = 37.0f / 378.0f;
    constexpr float c_2 = 0.0f;
    constexpr float c_3 = 250.0f / 621.0f;
    constexpr float c_4 = 125.0f / 594.0f;
    constexpr float c_5 = 0.0f;
    constexpr float c_6 = 512.0f / 1771.0f;

    constexpr float c_1s = 2825.0f / 27648.0f;
    constexpr float c_2s = 0.0f;
    constexpr float c_3s = 18575.0f / 48384.0f;
    constexpr float c_4s = 13525.0f / 55296.0f;
    constexpr float c_5s = 277.0f / 14336.0f;
    constexpr float c_6s = 0.25f;

    // Constants
    constexpr float grow_exponent = -0.2f;
    constexpr float shrink_exponent = -0.25f;
    constexpr float max_growth = 5.0f;
    constexpr float max_shrink = 0.1f;
    constexpr float safety = 0.9f;

    // Calculate Runge-Kutta coefficients
    float2 output_position;
    float2 used_delta_and_error;
    bool decreased = false;

    do
    {
        const float2 k1 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos));
        const float2 k2 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + b_21 * k1));
        const float2 k3 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + b_31 * k1 + b_32 * k2));
        const float2 k4 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + b_41 * k1 + b_42 * k2 + b_43 * k3));
        const float2 k5 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4));
        const float2 k6 = delta * sign * texture_interpolation<2>(textures[0], pos_to_texcoords(pos + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5));

        // Calculate error estimate
        const float2 fifth_order = pos + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
        const float2 fourth_order = pos + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

        const float2 difference = fabs(fifth_order - fourth_order);

        //const float2 scale = make_Real4(1.0);
        const float2 scale = fabs(texture_interpolation<2>(textures[0], pos_to_texcoords(pos)));

        const float error = fmaxf(0.0f, fmaxf(difference.x / scale.x, difference.y / scale.y)) / max_error;

        // Set new, adapted time step
        used_delta_and_error.x = delta;
        used_delta_and_error.y = error;

#if !(__streamlines_cuda_runge_kutta_45_fixed)
        if (error > 1.0f)
        {
            // Error too large, reduce time step
            delta *= fmaxf(max_shrink, safety * powf(error, shrink_exponent));
            decreased = true;
        }
        else
        {
            // Error (too) small, increase time step
            delta *= fminf(max_growth, safety * powf(error, grow_exponent));
            decreased = false;
        }
#endif

        // Set output
        output_position = fifth_order;
    }
    while (decreased);

    // Advect and store position
    pos = output_position;

    return used_delta_and_error;
}
#endif

/**
* Update label and distance
*
* @param label_conv_structure   ID of convergence structure
* @param dist_conv_structure    Distance between position and convergence structure
* @param label                  In/out (previous) label
* @param distance               In/out (previous) distance
*/
__device__
void update_label_and_dist(const int label_conv_structure, const float dist_conv_structure, short& label, float& distance)
{
    // Store new label and distance if distance to convergence structure is smaller than previous one
    if (dist_conv_structure < distance)
    {
        label = label_conv_structure;
        distance = dist_conv_structure;
    }
}

/**
* Update label and distance
*
* @param num_convergence_points Number of critical points
* @param num_convergence_lines  Number of segments (lines)
* @param pos                    Stream line position
* @param label                  In/out (previous) label
* @param distance               In/out (previous) distance
*/
__device__
void update_label_and_dist(const int num_convergence_points, const int num_convergence_lines, const float2 pos, short& label, float& distance)
{
#if __streamlines_cuda_shi_et_al
    label = -1;
    distance = 100000.0f;
#endif

    // Calculate distance of the current position to critical points
    for (int k = 0; k < num_convergence_points; ++k)
    {
        const float id = tex1Dfetch<float>(textures[3], k);

        const float2 p = make_real<float, 2>(tex1Dfetch<float2>(textures[2], k));
        const float dist = length(make_real<float, 2>(pos - p));

        update_label_and_dist(id, dist, label, distance);
    }

    // Calculate distance of the current position to line segments
    for (int k = 0; k < num_convergence_lines; ++k)
    {
        const float id = tex1Dfetch<float>(textures[5], k);

        const float2 p0 = make_real<float, 2>(tex1Dfetch<float2>(textures[4], k * 2 + 0));
        const float2 p1 = make_real<float, 2>(tex1Dfetch<float2>(textures[4], k * 2 + 1));
        const float dist = distance_point_line(make_real<float, 2>(pos), make_real<float, 2>(p0), make_real<float, 2>(p1));
        
        update_label_and_dist(id, dist, label, distance);
    }
}

/**
* Compute stream lines and update labels and distances
*
* @param num_convergence_points Number of critical points
* @param num_convergence_lines  Number of segments (lines)
* @param num_triangles          Number of triangles
* @param particles              Seed particles
* @param num_particles          Number of seed particles
* @param num_steps              Number of advection steps
* @param labels                 Output labels
* @param distances              Output distances
* @param integration_steps      Output integration step field
*/
__global__
void compute_streamlines_kernel(const int num_convergence_points, const int num_convergence_lines, const float sign,
    float2* particles, const int num_particles, const int num_steps, float* labels, float* distances, float* terminations
#if __streamlines_cuda_detailed_output
    , cudaSurfaceObject_t integration_steps
#endif
    )
{
    // Get kernel ID
    const int tid = threadIdx.x;
    const int gid = blockIdx.x*blockDim.x + tid;

    if (gid < num_particles && terminations[gid] == 0)
    {
        // Get initial values for labels, distances and positions
        short label = (short)labels[gid];
        float dist = distances[gid];
        short termination = (short)terminations[gid];
        float2 pos = particles[gid];

#if !(__streamlines_cuda_shi_et_al)
        // Initially update values by evaluating the distance to convergence structures
        update_label_and_dist(num_convergence_points, num_convergence_lines, pos, label, dist);
#endif

#if __streamlines_cuda_runge_kutta_45
        // Calculate initial time step
        float step = const_data[4].x * texture_interpolation<1>(textures[1], pos_to_texcoords(pos));
#endif

        for (int j = 0; j < num_steps; ++j)
        {
            // Advect using 4th-order Runge-Kutta
            const float2 posPrev = pos;
#if __streamlines_cuda_runge_kutta_45
#if __streamlines_cuda_detailed_output
            const float2 step_taken_and_error =
#endif
                advectRK45(pos, step, sign, const_data[5].x);
#else
            advectRK4(pos, const_data[4].x, sign);
#endif

#if !(__streamlines_cuda_shi_et_al)
            // Update values by evaluating the distance to convergence structures
            update_label_and_dist(num_convergence_points, num_convergence_lines, pos, label, dist);
#endif

#if __streamlines_cuda_detailed_output && __streamlines_cuda_runge_kutta_45
            // Update integration step size
            const int2 texPos = pos_to_surfcoords(posPrev);

            const float2 old_integration_step = surf3Dread<float2>(integration_steps, texPos.x * sizeof(float2), texPos.y, gid);
            
#if __streamlines_cuda_integration_steps_max
            const float integration_step = fmaxf(static_cast<float>(old_integration_step.x), step_taken_and_error.x);
#elif __streamlines_cuda_integration_steps_min
            const float integration_step = fminf(static_cast<float>(old_integration_step.x), step_taken_and_error.x);
#elif __streamlines_cuda_integration_steps_avg
            const float integration_step = static_cast<float>(old_integration_step.x) + step_taken_and_error.x;
#endif

#if __streamlines_cuda_integration_error_max
            const float integration_error = fmaxf(static_cast<float>(old_integration_step.y), step_taken_and_error.y);
#elif __streamlines_cuda_integration_error_min
            const float integration_error = fminf(static_cast<float>(old_integration_step.y), step_taken_and_error.y);
#elif __streamlines_cuda_integration_error_avg
            const float integration_error = static_cast<float>(old_integration_step.y) + step_taken_and_error.y;
#endif

            const float2 new_integration_step = make_real<float, 2>(integration_step, integration_error);
            surf3Dwrite(new_integration_step, integration_steps, texPos.x * sizeof(float2), texPos.y, gid);
#endif

            // If advection had no effect, abort the algorithm
            if (posPrev.x == pos.x && posPrev.y == pos.y)
            {
                termination = 2;
                break;
            }

            // If current position is outside of the domain, set "outside"-label and distance and
            // abort the algorithm
            const float2 pos_01 = (pos - const_data[0]) * const_data[1];

            if (pos_01.x < static_cast<float>(0.0) || pos_01.x > static_cast<float>(1.0) ||
                pos_01.y < static_cast<float>(0.0) || pos_01.y > static_cast<float>(1.0))
            {
                termination = 1;

#if __streamlines_cuda_shi_et_al
                label = -1;
                dist = 0.0;
#endif

                break;
            }
        }

#if __streamlines_cuda_shi_et_al
        // Update values by evaluating the distance to convergence structures at the final position
        update_label_and_dist(num_convergence_points, num_convergence_lines, num_triangles, pos, label, dist);
#endif

        // Store and return calculated values
        labels[gid] = label;
        distances[gid] = dist;
        terminations[gid] = termination;
        particles[gid] = pos;
    }
}

// ################################################################################################################################

namespace megamol
{
    namespace flowvis
    {
        static std::unique_ptr<streamlines_cuda_impl> impl = nullptr;

        streamlines_cuda::streamlines_cuda(const std::array<int, 2>& resolution, const std::array<float, 4>& domain,
            const std::vector<float>& vectors, const std::vector<float>& points, const std::vector<int>& point_ids,
            const std::vector<float>& lines, const std::vector<int>& line_ids, const float integration_timestep,
            const float max_integration_error)
        {
            impl = std::make_unique<streamlines_cuda_impl>(resolution, domain, vectors,
                points, point_ids, lines, line_ids, integration_timestep, max_integration_error);
        }

        void streamlines_cuda::update_labels(std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
            std::vector<float>& terminations, const int num_integration_steps, const float sign, const unsigned int num_particles_per_batch
#if __streamlines_cuda_detailed_output
            , std::vector<float>& integration_steps
#endif
        )
        {
            impl->update_labels(source, labels, distances, terminations, num_integration_steps, sign, num_particles_per_batch
#if __streamlines_cuda_detailed_output
                , integration_steps
#endif
            );
        }
    }
}

// ################################################################################################################################

namespace megamol
{
    namespace flowvis
    {
        streamlines_cuda_impl::streamlines_cuda_impl(const std::array<int, 2>& resolution, const std::array<float, 4>& domain,
            const std::vector<float>& vectors, const std::vector<float>& points, const std::vector<int>& point_ids,
            const std::vector<float>& lines, const std::vector<int>& line_ids, const float integration_timestep,
            const float max_integration_error)
            : resolution(resolution), d_velocity(nullptr), d_rk4_step(nullptr), d_convergence_points(nullptr), d_convergence_lines(nullptr), d_convergence_line_ids(nullptr)
        {
            // Get domain boundaries
            const std::size_t num_vectors = vectors.size() / 2;

            // Create constants and upload them to GPU
            const float2 domain_offset = make_real<float, 2>(domain[0], domain[2]);
            const float2 domain_scale = make_real<float, 2>(1.0f / (domain[1] - domain[0]), 1.0f / (domain[3] - domain[2]));
            const float2 texture_offset = make_real<float, 2>(-0.5f, -0.5f);
            const float2 texture_scale = make_real<float, 2>(this->resolution[0] - 1, this->resolution[1] - 1);
            const float2 time_scale = make_real<float, 2>(integration_timestep);
            const float2 max_error = make_real<float, 2>(max_integration_error);

            const std::array<float2, 6> h_const_data = { domain_offset, domain_scale, texture_offset, texture_scale, time_scale, max_error };

            cudaMemcpyToSymbol(const_data, h_const_data.data(), h_const_data.size() * sizeof(float2));

            // Create velocity texture and upload to GPU
            static_assert(sizeof(float2) == 2 * sizeof(float), "CUDA type 'float2' must only consist of two 'float' members.");

            initialize_texture(vectors.data(), 2, &this->velocity_texture, &this->d_velocity);
            cudaMemcpyToSymbol(textures, &this->velocity_texture, sizeof(cudaTextureObject_t));

            // Create texture for Runge-Kutta step size and upload to GPU
            std::vector<float> h_rk4step(num_vectors);

            const float cellx = (domain[1] - domain[0]) / (this->resolution[0] - 1);
            const float celly = (domain[3] - domain[2]) / (this->resolution[1] - 1);
            const float cell_diag = std::sqrt(cellx * cellx + celly * celly);

            for (int i = 0; i < num_vectors; ++i)
            {
                h_rk4step[i] = cell_diag;
            }

            initialize_texture(h_rk4step.data(), 1, &this->rk4_step_texture, &this->d_rk4_step);
            cudaMemcpyToSymbol(textures, &this->rk4_step_texture, sizeof(cudaTextureObject_t), sizeof(cudaTextureObject_t));

            // Create texture for critical points and upload to GPU
            this->num_convergence_points = static_cast<int>(point_ids.size());

            if (!points.empty())
            {
                std::vector<float2> h_convergence_points(this->num_convergence_points);
                std::vector<float> h_ids(this->num_convergence_points);

                for (int i = 0; i < this->num_convergence_points; ++i)
                {
                    h_ids[i] = static_cast<float>(point_ids[i]);
                    h_convergence_points[i] = make_float2(points[i * 2 + 0], points[i * 2 + 1]);
                }

                initialize_texture((void*)h_convergence_points.data(), this->num_convergence_points,
                    sizeof(float), sizeof(float), 0, 0, &this->convergence_points_texture, (void**)&this->d_convergence_points);

                initialize_texture((void*)h_ids.data(), this->num_convergence_points,
                    sizeof(float), 0, 0, 0, &this->convergence_point_ids_texture, (void**)&this->d_convergence_point_ids);

                cudaMemcpyToSymbol(textures, &this->convergence_points_texture, sizeof(cudaTextureObject_t), 2 * sizeof(cudaTextureObject_t));
                cudaMemcpyToSymbol(textures, &this->convergence_point_ids_texture, sizeof(cudaTextureObject_t), 3 * sizeof(cudaTextureObject_t));
            }

            // Create texture for line segments and upload to GPU
            this->num_convergence_lines = static_cast<int>(line_ids.size());

            if (!lines.empty())
            {
                std::vector<float2> h_convergence_lines(this->num_convergence_lines * 2);
                std::vector<float> h_ids(this->num_convergence_lines);

                for (int i = 0; i < this->num_convergence_lines; ++i)
                {
                    h_ids[i] = static_cast<float>(line_ids[i]);
                    h_convergence_lines[i * 2 + 0] = make_float2(lines[i * 4 + 0], lines[i * 4 + 1]);
                    h_convergence_lines[i * 2 + 1] = make_float2(lines[i * 4 + 2], lines[i * 4 + 3]);
                }

                initialize_texture((void*)h_convergence_lines.data(), this->num_convergence_lines * 2,
                    sizeof(float), sizeof(float), 0, 0, &this->convergence_lines_texture, (void**)&this->d_convergence_lines);

                initialize_texture((void*)h_ids.data(), this->num_convergence_lines,
                    sizeof(float), 0, 0, 0, &this->convergence_line_ids_texture, (void**)&this->d_convergence_line_ids);

                cudaMemcpyToSymbol(textures, &this->convergence_lines_texture, sizeof(cudaTextureObject_t), 4 * sizeof(cudaTextureObject_t));
                cudaMemcpyToSymbol(textures, &this->convergence_line_ids_texture, sizeof(cudaTextureObject_t), 5 * sizeof(cudaTextureObject_t));
            }
        }

        streamlines_cuda_impl::~streamlines_cuda_impl()
        {
            if (this->d_velocity)
            {
                cudaDestroyTextureObject(this->velocity_texture);
                cudaFreeArray(this->d_velocity);
            }

            if (this->d_rk4_step)
            {
                cudaDestroyTextureObject(this->rk4_step_texture);
                cudaFreeArray(this->d_rk4_step);
            }

            if (this->d_convergence_points)
            {
                cudaDestroyTextureObject(this->convergence_points_texture);
                cudaFree(this->d_convergence_points);
                cudaDestroyTextureObject(this->convergence_point_ids_texture);
                cudaFree(this->d_convergence_point_ids);
            }

            if (this->d_convergence_lines)
            {
                cudaDestroyTextureObject(this->convergence_lines_texture);
                cudaFree(this->d_convergence_lines);
                cudaDestroyTextureObject(this->convergence_line_ids_texture);
                cudaFree(this->d_convergence_line_ids);
            }
        }

        void streamlines_cuda_impl::update_labels(std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
            std::vector<float>& terminations, const int num_integration_steps, const float sign, unsigned int num_particles_per_batch
#if __streamlines_cuda_detailed_output
            , std::vector<float>& integration_steps
#endif
        )
        {
            // Subdivide the input
            const unsigned int num_particles = static_cast<unsigned int>(source.size() / 2);

#if __streamlines_cuda_detailed_output
            num_particles_per_batch = std::min(2048, num_particles_per_batch);
#endif

            cudaError_t err;
            std::stringstream ss;

            const std::size_t max_num_particles_per_batch = std::min(num_particles_per_batch, num_particles);

            // Allocate memory on CPU
#if __streamlines_cuda_detailed_output
            std::vector<float2> h_integration_steps(this->resolution[0] * this->resolution[1] * max_num_particles_per_batch);
#endif

            // Allocate memory
            float *d_labels;
            err = cudaMalloc((void**)&d_labels, max_num_particles_per_batch * sizeof(float));
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for labels." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            float *d_dists;
            err = cudaMalloc((void**)&d_dists, max_num_particles_per_batch * sizeof(float));
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for distances." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            float *d_terminations;
            err = cudaMalloc((void**)&d_terminations, max_num_particles_per_batch * sizeof(float));
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            float2 *d_particles;
            err = cudaMalloc((void**)&d_particles, max_num_particles_per_batch * sizeof(float2));
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for particles." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

#if __streamlines_cuda_detailed_output
            const cudaExtent extent = { static_cast<std::size_t>(this->resolution[0]), static_cast<std::size_t>(this->resolution[1]), max_num_particles_per_batch };
            const cudaChannelFormatDesc desc = cudaCreateChannelDesc(sizeof(float) * 8, sizeof(float) * 8, 0, 0, cudaChannelFormatKindSigned);

            cudaArray_t d_integration_steps;
            err = cudaMalloc3DArray(&d_integration_steps, &desc, extent, cudaArraySurfaceLoadStore);
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc3DArray for integration steps fields." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            // Create surface object
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = d_integration_steps;

            cudaSurfaceObject_t integration_steps = 0;
            cudaCreateSurfaceObject(&integration_steps, &resDesc);

            // Create copy parameters
            cudaPitchedPtr cpuPtr;
            cpuPtr.ptr = h_integration_steps.data();
            cpuPtr.pitch = this->resolution[0] * sizeof(float2);
            cpuPtr.xsize = this->resolution[0];
            cpuPtr.ysize = this->resolution[1];

            cudaMemcpy3DParms toGPUParams = { 0 };
            toGPUParams.srcPtr = cpuPtr;
            toGPUParams.dstArray = d_integration_steps;
            toGPUParams.extent = extent;
            toGPUParams.kind = cudaMemcpyHostToDevice;

            cudaMemcpy3DParms fromGPUParams = { 0 };
            fromGPUParams.dstPtr = cpuPtr;
            fromGPUParams.srcArray = d_integration_steps;
            fromGPUParams.extent = extent;
            fromGPUParams.kind = cudaMemcpyDeviceToHost;
#endif

            for (unsigned int offset = 0; offset < num_particles; offset += num_particles_per_batch)
            {
                const int num_particles_this_batch = std::min(num_particles_per_batch, num_particles - offset);

                // Copy data to GPU memory
#if __streamlines_cuda_detailed_output
#if __streamlines_cuda_integration_steps_max
                std::fill(h_integration_steps.begin(), h_integration_steps.end(), make_real<float, 2>(0.0));
#elif __streamlines_cuda_integration_steps_min
                std::fill(h_integration_steps.begin(), h_integration_steps.end(), make_real<float, 2>(std::numeric_limits<float>::max()));
#elif __streamlines_cuda_integration_steps_avg
                std::fill(h_integration_steps.begin(), h_integration_steps.end(), make_real<float, 2>(0.0));
#endif
#endif

                err = cudaMemcpy(d_labels, &labels[offset], num_particles_this_batch * sizeof(float), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for labels." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_dists, &distances[offset], num_particles_this_batch * sizeof(float), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for distances." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_terminations, &terminations[offset], num_particles_this_batch * sizeof(float), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_particles, &source[2 * offset], num_particles_this_batch * sizeof(float2), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for particles." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

#if __streamlines_cuda_detailed_output
                err = cudaMemcpy3D(&toGPUParams);
                if (err)
                {
                    ss << "Error copyingto GPU memory using cudaMemcpy3D for integration steps fields." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }
#endif

                //--------------------------------------------------------------------------

                compute_streamlines(d_particles, num_particles_this_batch, this->num_convergence_points, this->num_convergence_lines,
                    num_integration_steps, sign, d_labels, d_dists, d_terminations
#if __streamlines_cuda_detailed_output    
                    , integration_steps
#endif
                );
                
                //--------------------------------------------------------------------------

                // Copy data from GPU memory
                err = cudaMemcpy(&labels[offset], d_labels, num_particles_this_batch * sizeof(float), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for labels." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(&distances[offset], d_dists, num_particles_this_batch * sizeof(float), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for distances." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(&terminations[offset], d_terminations, num_particles_this_batch * sizeof(float), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(&source[2 * offset], d_particles, num_particles_this_batch * sizeof(float2), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for particles." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

#if __streamlines_cuda_detailed_output
                err = cudaMemcpy3D(&fromGPUParams);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy3D for integration steps fields." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }
#endif

#if __streamlines_cuda_detailed_output
                // Store information in output variables
                for (int i = 0; i < num_particles_this_batch; ++i)
                {
                    // Calculate aggregated integration steps field
                    for (int y = 0; y < this->resolution[1]; ++y)
                    {
                        for (int x = 0; x < this->resolution[0]; ++x)
                        {
                            const std::size_t field_index = x + this->resolution[0] * (y + this->resolution[1] * i);
                            const vtkIdType map_index = x + this->resolution[0] * y;

#if __streamlines_cuda_integration_steps_max
                            integration_steps_field->SetComponent(map_index, 0,
                                std::max(integration_steps_field->GetComponent(map_index, 0), static_cast<double>(h_integration_steps[field_index].x)));
#elif __streamlines_cuda_integration_steps_min
                            integration_steps_field->SetComponent(map_index, 0,
                                std::min(integration_steps_field->GetComponent(map_index, 0), static_cast<double>(h_integration_steps[field_index].x)));
#elif __streamlines_cuda_integration_steps_avg
                            integration_steps_field->SetComponent(map_index, 0,
                                (integration_steps_field->GetComponent(map_index, 0) + static_cast<double>(h_integration_steps[field_index].x / num_steps)) / 2.0);
#endif

#if __streamlines_cuda_integration_error_max
                            integration_steps_field->SetComponent(map_index, 1,
                                std::max(integration_steps_field->GetComponent(map_index, 1), static_cast<double>(h_integration_steps[field_index].y)));
#elif __streamlines_cuda_integration_error_min
                            integration_steps_field->SetComponent(map_index, 1,
                                std::min(integration_steps_field->GetComponent(map_index, 1), static_cast<double>(h_integration_steps[field_index].y)));
#elif __streamlines_cuda_integration_error_avg
                            integration_steps_field->SetComponent(map_index, 1,
                                (integration_steps_field->GetComponent(map_index, 1) + static_cast<double>(h_integration_steps[field_index].y / num_steps)) / 2.0);
#endif
                        }
                    }
                }
#endif
            }

            cudaFree(d_labels);
            cudaFree(d_dists);
            cudaFree(d_terminations);
            cudaFree(d_particles);

#if __streamlines_cuda_detailed_output
            cudaDestroySurfaceObject(integration_steps);

            cudaFree(d_integration_steps);
#endif
        }

        void streamlines_cuda_impl::compute_streamlines(float2* d_particles, const int num_particles, const int num_convergence_points, const int num_convergence_lines,
            const int num_steps, const float sign, float* d_labels, float* d_dists, float* d_terminations
#if __streamlines_cuda_detailed_output
            , cudaSurfaceObject_t integration_steps
#endif
        )
        {
            // Run CUDA kernel
            int num_threads = 64;
            int num_blocks = num_particles / num_threads + (num_particles % num_threads == 0 ? 0 : 1);

            compute_streamlines_kernel __cuda_kernel_start(num_blocks, num_threads) (num_convergence_points, num_convergence_lines, sign,
                d_particles, num_particles, num_steps, d_labels, d_dists, d_terminations
#if __streamlines_cuda_detailed_output
                , integration_steps
#endif
                );
        }

        void streamlines_cuda_impl::initialize_texture(const void* h_data, const int num_components, cudaTextureObject_t* texture, cudaArray** d_data)
        {
            const cudaExtent extent = { static_cast<std::size_t>(this->resolution[0]), static_cast<std::size_t>(this->resolution[1]), 0 };
            cudaChannelFormatDesc desc = cudaCreateChannelDesc(sizeof(float) * 8, num_components > 1 ? sizeof(float) * 8 : 0,
                num_components > 2 ? sizeof(float) * 8 : 0, num_components > 3 ? sizeof(float) * 8 : 0, cudaChannelFormatKindFloat);

            cudaError_t err;
            std::stringstream ss;

            err = cudaMalloc3DArray(d_data, &desc, extent);
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc3DArray for velocity or RK4 step size." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            cudaPitchedPtr srcPtr;
            srcPtr.ptr = const_cast<void*>(h_data);
            srcPtr.pitch = static_cast<std::size_t>(this->resolution[0] * sizeof(float) * num_components);
            srcPtr.xsize = static_cast<std::size_t>(this->resolution[0]);
            srcPtr.ysize = static_cast<std::size_t>(this->resolution[1]);

            cudaMemcpy3DParms params = { 0 };
            params.srcPtr = srcPtr;
            params.dstArray = *d_data;
            params.extent = extent;
            params.kind = cudaMemcpyHostToDevice;

            err = cudaMemcpy3D(&params);
            if (err)
            {
                ss << "Error copying memory using cudaMemcpy3D for velocity or RK4 step size." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = *d_data;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeBorder;
            texDesc.addressMode[1] = cudaAddressModeBorder;
            texDesc.addressMode[2] = cudaAddressModeBorder;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(texture, &resDesc, &texDesc, nullptr);
        }

        void streamlines_cuda_impl::initialize_texture(const void* h_data, const int num_elements, const int c0,
            const int c1, const int c2, const int c3, cudaTextureObject_t* texture, void** d_data)
        {
            const int num_bytes = c0 + c1 + c2 + c3;

            cudaError_t err;
            std::stringstream ss;

            err = cudaMalloc((void**)d_data, num_elements * num_bytes);
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for convergence structures." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            err = cudaMemcpy(*d_data, h_data, num_elements * num_bytes, cudaMemcpyHostToDevice);
            if (err)
            {
                ss << "Error copying memory using cudaMemcpy for convergence structures." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            cudaChannelFormatDesc cfdesc = cudaCreateChannelDesc(c0 * 8, c1 * 8, c2 * 8, c3 * 8, cudaChannelFormatKindFloat);

            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeLinear;
            resDesc.res.linear.devPtr = *d_data;
            resDesc.res.linear.desc = cfdesc;
            resDesc.res.linear.sizeInBytes = num_elements * num_bytes;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.addressMode[0] = cudaAddressModeBorder;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(texture, &resDesc, &texDesc, nullptr);
        }
    }
}