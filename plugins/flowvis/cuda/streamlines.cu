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

// Transformations: domain_offset, domain_scale, texture_offset, texture_scale, time_scale, max_integration_error}
__constant__ float4 constData[6];
// Textures: vector field, 4th-order runge-kutta step size, crit points, lines, line ids, triangles, triangle ids
__constant__ cudaTextureObject_t textures[8];

/**
* Transform world position to texture coordinates
*
* @param pos World position
*
* @return Texture coordinates
*/
inline __device__
float4 pos_to_texcoords(const float4 pos)
{
    // Transform position from [physical] to [0 : 1]
    const float4 scaled_position = (pos - constData[0]) * constData[1];

    // Transform position from [0 : 1] to [0.5 : texWidth - 0.5]
    return scaled_position * constData[3] - constData[2];
}

/**
* Transform world position to surface memory coordinates
*
* @param pos World position
*
* @return Surface memory coordinates
*/
inline __device__
int2 pos_to_surfcoords(const float4 pos)
{
    // Transform position from [physical] to [0 : 1]
    float4 scaled_position = (pos - constData[0]) * constData[1];

    scaled_position = fmaxf(fminf(scaled_position, make_real<float, 4>(1.0)), make_real<float, 4>(0.0));

    // Transform position from [0 : 1] to [0 : texWidth - 1]
    return make_int2(floor(scaled_position.x * constData[3].x), floor(scaled_position.y * constData[3].y));
}

#if !(__streamlines_cuda_runge_kutta_45)
/**
* Advect position using 4th-order Runge-Kutta
*
* @param pos In/out position
* @param delta Time step coefficient
* @param sign Sign (1: forward, 2: backward integration)
*/
__device__
void advectRK4(float4& pos, const float delta, const float sign)
{
    // Calculate Runge-Kutta coefficients
    float4 velocity, scaled_position;

    const float step = tex3D_interp<1>(textures[1], pos_to_texcoords(pos));

    const float4 k1 = step * delta * normalizeSafe(sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos)));
    const float4 k2 = step * delta * normalizeSafe(sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + static_cast<float>(0.5) * k1)));
    const float4 k3 = step * delta * normalizeSafe(sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + static_cast<float>(0.5) * k2)));
    const float4 k4 = step * delta * normalizeSafe(sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + k3)));

    // Advect and store position
    pos += (static_cast<float>(1.0) / static_cast<float>(6.0)) * (k1 + static_cast<float>(2.0) * k2 + static_cast<float>(2.0) * k3 + k4);
}
#else
/**
* Advect position using 4th-order Runge-Kutta with 5th-order error estimation for adaptive time steps
* See http://www.aip.de/groups/soe/local/numres/bookcpdf/c16-2.pdf for details
*
* @param pos In/out position
* @param delta Time step coefficient
* @param sign Sign (1: forward, 2: backward integration)
* @param max_error Maximum error allowed, used for step size adjustment
*
* @return Time step coefficient based on error estimation, with corresponding error
*/
__device__
float2 advectRK45(float4& pos, float& delta, const float sign, const float max_error)
{
    // Cash-Karp parameters
    constexpr float b_21 = 0.2;
    constexpr float b_31 = 0.075;
    constexpr float b_41 = 0.3;
    constexpr float b_51 = -11.0 / 54.0;
    constexpr float b_61 = 1631.0 / 55296.0;
    constexpr float b_32 = 0.225;
    constexpr float b_42 = -0.9;
    constexpr float b_52 = 2.5;
    constexpr float b_62 = 175.0 / 512.0;
    constexpr float b_43 = 1.2;
    constexpr float b_53 = -70.0 / 27.0;
    constexpr float b_63 = 575.0 / 13824.0;
    constexpr float b_54 = 35.0 / 27.0;
    constexpr float b_64 = 44275.0 / 110592.0;
    constexpr float b_65 = 253.0 / 4096.0;

    constexpr float c_1 = 37.0 / 378.0;
    constexpr float c_2 = 0.0;
    constexpr float c_3 = 250.0 / 621.0;
    constexpr float c_4 = 125.0 / 594.0;
    constexpr float c_5 = 0.0;
    constexpr float c_6 = 512.0 / 1771.0;

    constexpr float c_1s = 2825.0 / 27648.0;
    constexpr float c_2s = 0.0;
    constexpr float c_3s = 18575.0 / 48384.0;
    constexpr float c_4s = 13525.0 / 55296.0;
    constexpr float c_5s = 277.0 / 14336.0;
    constexpr float c_6s = 0.25;

    // Constants
    constexpr float grow_exponent = -0.2;
    constexpr float shrink_exponent = -0.25;
    constexpr float max_growth = 5.0;
    constexpr float max_shrink = 0.1;
    constexpr float safety = 0.9;

    // Calculate Runge-Kutta coefficients
    float4 output_position;
    float2 used_delta_and_error;
    bool decreased = false;

    do
    {
        const float4 k1 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos));
        const float4 k2 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + b_21 * k1));
        const float4 k3 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + b_31 * k1 + b_32 * k2));
        const float4 k4 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + b_41 * k1 + b_42 * k2 + b_43 * k3));
        const float4 k5 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4));
        const float4 k6 = delta * sign * tex3D_interp<4>(textures[0], pos_to_texcoords(pos + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5));

        // Calculate error estimate
        const float4 fifth_order = pos + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
        const float4 fourth_order = pos + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

        const float4 difference = fabs(fifth_order - fourth_order);

        //const float4 scale = make_Real4(1.0);
        const float4 scale = fabs(tex3D_interp<4>(textures[0], pos_to_texcoords(pos)));

        const float error = fmaxf(static_cast<float>(0.0), fmaxf(difference.x / scale.x, fmaxf(difference.y / scale.y, difference.z / scale.z))) / max_error;

        // Set new, adapted time step
        used_delta_and_error.x = delta;
        used_delta_and_error.y = error;

#if !(__streamlines_cuda_runge_kutta_45_fixed)
        if (error > static_cast<float>(1.0))
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
* @param label_conv_structure ID of convergence structure
* @param dist_conv_structure Distance between position and convergence structure
* @param label In/out (previous) label
* @param distance In/out (previous) distance
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
* @param num_critical_points Number of critical points
* @param num_segments Number of segments (lines)
* @param num_triangles Number of triangles
* @param pos Stream line position
* @param label In/out (previous) label
* @param distance In/out (previous) distance
*/
__device__
void update_label_and_dist(const int num_critical_points, const int num_segments, const int num_triangles, const float4 pos, short& label, float& distance)
{
#if __streamlines_cuda_shi_et_al
    label = -1;
    distance = 100000.0f;
#endif

    // Calculate distance of the current position to critical points
    for (int k = 0; k < num_critical_points; ++k)
    {
        const float id = static_cast<float>(tex1Dfetch<float>(textures[3], k));

        const float4 p = make_real<float, 4, float>(tex1Dfetch<float4>(textures[2], k));
        const float dist = length(make_real<float, 3>(pos - p));

        update_label_and_dist(id, dist, label, distance);
    }

    // Calculate distance of the current position to line segments
    for (int k = 0; k < num_segments; ++k)
    {
        const float id = static_cast<float>(tex1Dfetch<float>(textures[5], k));

        const float4 p0 = make_real<float, 4, float>(tex1Dfetch<float4>(textures[4], k * 2 + 0));
        const float4 p1 = make_real<float, 4, float>(tex1Dfetch<float4>(textures[4], k * 2 + 1));
        const float dist = distance_point_line(make_real<float, 3>(pos), make_real<float, 3>(p0), make_real<float, 3>(p1));
        
        update_label_and_dist(id, dist, label, distance);
    }

    // Calculate distance of the current position to triangles
    for (int k = 0; k < num_triangles; ++k)
    {
        const float id = static_cast<float>(tex1Dfetch<float>(textures[7], k));

        const float4 p0 = make_real<float, 4, float>(tex1Dfetch<float4>(textures[6], k * 3 + 0));
        const float4 p1 = make_real<float, 4, float>(tex1Dfetch<float4>(textures[6], k * 3 + 1));
        const float4 p2 = make_real<float, 4, float>(tex1Dfetch<float4>(textures[6], k * 3 + 2));
        const float dist = distance_point_triangle(make_real<float, 3>(pos), make_real<float, 3>(p0), make_real<float, 3>(p1), make_real<float, 3>(p2));

        update_label_and_dist(id, dist, label, distance);
    }
}

/**
* Compute stream lines and update labels and distances
*
* @param num_critical_points Number of critical points
* @param num_segments Number of segments (lines)
* @param num_triangles Number of triangles
* @param particles Seed particles
* @param num_particles Number of seed particles
* @param num_steps Number of advection steps
* @param labels Output labels
* @param distances Output distances
* @param integration_steps Output integration step field
*/
__global__
void compute_streamlines_kernel(const int num_critical_points, const int num_segments, const int num_triangles, const float sign,
    float4* particles, const int num_particles, const int num_steps, short* labels, float* distances, short* terminations
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
        short label = labels[gid];
        float dist = distances[gid];
        short termination = terminations[gid];
        float4 pos = particles[gid];

#if !(__streamlines_cuda_shi_et_al)
        // Initially update values by evaluating the distance to convergence structures
        update_label_and_dist(num_critical_points, num_segments, num_triangles, pos, label, dist);
#endif

#if __streamlines_cuda_runge_kutta_45
        // Calculate initial time step
        float step = constData[4].x * tex3D_interp<1>(textures[1], pos_to_texcoords(pos));
#endif

        for (int j = 0; j < num_steps; ++j)
        {
            // Advect using 4th-order Runge-Kutta
            const float4 posPrev = pos;
#if __streamlines_cuda_runge_kutta_45
#if __streamlines_cuda_detailed_output
            const float2 step_taken_and_error =
#endif
                advectRK45(pos, step, sign, constData[5].x);
#else
            advectRK4(pos, constData[4].x, sign);
#endif

#if !(__streamlines_cuda_shi_et_al)
            // Update values by evaluating the distance to convergence structures
            update_label_and_dist(num_critical_points, num_segments, num_triangles, pos, label, dist);
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
            if (posPrev.x == pos.x && posPrev.y == pos.y && posPrev.z == pos.z)
            {
                termination = 2;
                break;
            }

            // If current position is outside of the domain, set "outside"-label and distance and
            // abort the algorithm
            const float4 pos_01 = (pos - constData[0]) * constData[1];

            if (pos_01.x < static_cast<float>(0.0) || pos_01.x > static_cast<float>(1.0) ||
                pos_01.y < static_cast<float>(0.0) || pos_01.y > static_cast<float>(1.0) /*||
                pos_01.z < static_cast<float>(0.0) || pos_01.z > static_cast<float>(1.0)*/) // TODO
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
        update_label_and_dist(num_critical_points, num_segments, num_triangles, pos, label, dist);
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
        streamlines_cuda::streamlines_cuda(const std::vector<float>& positions, const std::vector<float>& vectors, const std::vector<float>& points,
            const std::vector<int>& point_ids, const std::vector<float>& lines, const std::vector<int>& line_ids,
            float integration_timestep, float max_integration_error)
        {
            this->impl = std::make_unique<streamlines_cuda_impl>(positions, vectors,
                points, point_ids, lines, line_ids, integration_timestep, max_integration_error);
        }

        streamlines_cuda::~streamlines_cuda()
        {
        }

        void streamlines_cuda::update_labels(const std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
            std::vector<float>& end_positions, int num_integration_steps, float sign
#if __streamlines_cuda_detailed_output
            , std::vector<float>& integration_steps
#endif
        )
        {
            this->impl->update_labels(source, labels, distances, end_positions, num_integration_steps, sign
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
        streamlines_cuda_impl::streamlines_cuda_impl(const std::vector<float>& positions, const std::vector<float>& vectors, const std::vector<float>& points,
            const std::vector<int>& point_ids, const std::vector<float>& lines, const std::vector<int>& line_ids,
            const float integration_timestep, const float max_integration_error) :
            d_velocity(nullptr), d_rk4_step(nullptr), d_critical_points(nullptr), d_segments(nullptr),
            d_segment_ids(nullptr), d_triangles(nullptr), d_triangleIds(nullptr)
        {
            // Get resolution and number of initial points
            velocities->GetDimensions(this->resolution.data());

            if (this->resolution[0] <= 0) this->resolution[0] = 1;
            if (this->resolution[1] <= 0) this->resolution[1] = 1;
            if (this->resolution[2] <= 0) this->resolution[2] = 1;

            std::array<double, 6> bounds;
            velocities->GetBounds(bounds.data());

            const int num_tuples = this->resolution[0] * this->resolution[1] * this->resolution[2];

            // Create constants and upload them to GPU
            const float4 domain_offset = make_real<float, 4>(bounds[0], bounds[2], bounds[4], 0.0);
            const float4 domain_scale = make_real<float, 4>(1.0 / (bounds[1] - bounds[0]), 1.0 / (bounds[3] - bounds[2]),
                (velocities->GetDataDimension() == 2) ? 1.0 : (1.0 / (bounds[5] - bounds[4])), 1.0);
            const float4 texture_offset = make_real<float, 4>(-0.5, -0.5, -0.5, 0.0);
            const float4 texture_scale = make_real<float, 4>(this->resolution[0] - 1, this->resolution[1] - 1,
                (velocities->GetDataDimension() == 2) ? 1 : (this->resolution[2] - 1), 1);
            const float4 time_scale = make_real<float, 4>(integration_timestep);
            const float4 max_error = make_real<float, 4>(max_integration_error);

            const std::array<float4, 6> h_constData = { domain_offset, domain_scale, texture_offset, texture_scale, time_scale, max_error };

            cudaMemcpyToSymbol(constData, h_constData.data(), h_constData.size() * sizeof(float4));

            // Create velocity texture and upload to GPU
            vtkDataArray *velocityArray = velocities->GetPointData()->GetArray("Data");

            std::vector<float4> h_velocity(num_tuples);

            for (int i = 0; i < num_tuples; ++i)
            {
                h_velocity[i] = make_float4(
                    static_cast<float>(velocityArray->GetComponent(i, 0)),
                    static_cast<float>(velocityArray->GetComponent(i, 1)),
                    static_cast<float>(velocityArray->GetComponent(i, 2)),
                    0.0f);
            }

            initialize_texture(h_velocity.data(), 4, &this->velocity_texture, &this->d_velocity);
            cudaMemcpyToSymbol(textures, &this->velocity_texture, sizeof(cudaTextureObject_t));

            // Create texture for Runge-Kutta step size and upload to GPU
            std::vector<float> h_rk4step(num_tuples);

            const float cellx = (bounds[1] - bounds[0]) / (this->resolution[0] - 1);
            const float celly = (bounds[3] - bounds[2]) / (this->resolution[1] - 1);
            const float cellz = (velocities->GetDataDimension() == 2) ? static_cast<float>(0.0) : ((bounds[5] - bounds[4]) / (this->resolution[2] - 1));
            const auto cell_diag = static_cast<float>(std::sqrt(cellx * cellx + celly * celly + cellz * cellz));

            for (int i = 0; i < num_tuples; ++i)
            {
                h_rk4step[i] = cell_diag;
            }

            initialize_texture(h_rk4step.data(), 1, &this->rk4_step_texture, &this->d_rk4_step);
            cudaMemcpyToSymbol(textures, &this->rk4_step_texture, sizeof(cudaTextureObject_t), sizeof(cudaTextureObject_t));

            // Get points and IDs of convergence structures
            vtkPoints *feature_points = convergence_structures->GetPoints();
            vtkDataArray *ids = convergence_structures->GetCellData()->GetArray("Ids");

            std::cout << "IDs provided = " << ((ids != nullptr) ? "yes" : "no") << std::endl;

            // Create texture for critical points and upload to GPU
            vtkCellArray *critical_points = convergence_structures->GetVerts();
            this->num_critical_points = critical_points->GetNumberOfCells();

            if (this->num_critical_points > 0)
            {
                std::vector<float4> h_critPoints(this->num_critical_points);
                std::vector<float> h_ids(this->num_critical_points);

                int idx = 0;
                critical_points->InitTraversal();

                for (int i = 0; i < this->num_critical_points; ++i)
                {
                    vtkIdType *pts, npts;
                    critical_points->GetNextCell(npts, pts);

                    std::array<double, 3> p;
                    feature_points->GetPoint(pts[0], p.data());

                    h_ids[idx++] = static_cast<float>((ids != nullptr) ? ids->GetComponent(i, 0) : i);
                    h_critPoints[i] = make_float4(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]), 1.0f);
                }

                initialize_texture((void*)h_critPoints.data(), this->num_critical_points, 4, 4, 4, 4, &this->critical_points_texture, (void**)&this->d_critical_points);
                initialize_texture((void*)h_ids.data(), this->num_critical_points, 4, 0, 0, 0, &this->critical_point_ids_texture, (void**)&this->d_critical_point_ids);

                cudaMemcpyToSymbol(textures, &this->critical_points_texture, sizeof(cudaTextureObject_t), 2 * sizeof(cudaTextureObject_t));
                cudaMemcpyToSymbol(textures, &this->critical_point_ids_texture, sizeof(cudaTextureObject_t), 3 * sizeof(cudaTextureObject_t));
            }

            // Create texture for line segments and upload to GPU
            vtkCellArray *line_cells = convergence_structures->GetLines();
            int num_line_segment_cells = line_cells->GetNumberOfCells();

            this->num_line_segments = 0;
            line_cells->InitTraversal();

            for (int i = 0; i < num_line_segment_cells; ++i)
            {
                vtkIdType *pts, npts;
                line_cells->GetNextCell(npts, pts);

                this->num_line_segments += npts - 1;
            }

            if (this->num_line_segments > 0)
            {
                std::vector<float4> h_segments(this->num_line_segments * 2);
                std::vector<float> h_ids(this->num_line_segments);

                int idx = 0;
                line_cells->InitTraversal();

                for (int i = 0; i < num_line_segment_cells; ++i)
                {
                    vtkIdType *pts, npts;
                    line_cells->GetNextCell(npts, pts);

                    for (int j = 0; j < npts - 1; ++j)
                    {
                        std::array<double, 3> p0, p1;
                        feature_points->GetPoint(pts[j], p0.data());
                        feature_points->GetPoint(pts[j + 1], p1.data());

                        h_ids[idx / 2] = static_cast<float>((ids != nullptr) ? ids->GetComponent(i + this->num_critical_points, 0) : i + this->num_critical_points);

                        h_segments[idx++] = make_float4(static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2]), 1.0f);
                        h_segments[idx++] = make_float4(static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]), 1.0f);
                    }
                }

                initialize_texture((void*)h_segments.data(), this->num_line_segments * 2, 4, 4, 4, 4, &this->segments_texture, (void**)&this->d_segments);
                initialize_texture((void*)h_ids.data(), this->num_line_segments, 4, 0, 0, 0, &this->segment_ids_texture, (void**)&this->d_segment_ids);

                cudaMemcpyToSymbol(textures, &this->segments_texture, sizeof(cudaTextureObject_t), 4 * sizeof(cudaTextureObject_t));
                cudaMemcpyToSymbol(textures, &this->segment_ids_texture, sizeof(cudaTextureObject_t), 5 * sizeof(cudaTextureObject_t));
            }

            // Create texture for triangles and upload to GPU
            vtkCellArray *triangle_cells = convergence_structures->GetPolys();
            int num_triangle_cells = triangle_cells->GetNumberOfCells();

            this->num_triangles = 0;
            triangle_cells->InitTraversal();

            for (int i = 0; i < num_triangle_cells; ++i)
            {
                vtkIdType *pts, npts;
                triangle_cells->GetNextCell(npts, pts);

                if (npts == 3)
                {
                    ++this->num_triangles;
                }
            }

            if (this->num_triangles > 0)
            {
                std::vector<float4> h_triangles(this->num_triangles * 3);
                std::vector<float> h_ids(this->num_triangles);

                int idx = 0;
                triangle_cells->InitTraversal();

                for (int i = 0; i < num_triangle_cells; ++i)
                {
                    vtkIdType *pts, npts;
                    triangle_cells->GetNextCell(npts, pts);

                    if (npts == 3)
                    {
                        std::array<double, 3> p0, p1, p2;
                        feature_points->GetPoint(pts[0], p0.data());
                        feature_points->GetPoint(pts[1], p1.data());
                        feature_points->GetPoint(pts[2], p2.data());

                        h_ids[idx / 3] = static_cast<float>((ids != nullptr) ? ids->GetComponent(i + this->num_critical_points + this->num_line_segments, 0)
                            : i + this->num_critical_points + this->num_line_segments);

                        h_triangles[idx++] = make_float4(static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2]), 1.0f);
                        h_triangles[idx++] = make_float4(static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]), 1.0f);
                        h_triangles[idx++] = make_float4(static_cast<float>(p2[0]), static_cast<float>(p2[1]), static_cast<float>(p2[2]), 1.0f);
                    }
                }

                initialize_texture((void*)h_triangles.data(), this->num_triangles * 3, 4, 4, 4, 4, &trianglesTex, (void**)&d_triangles);
                initialize_texture((void*)h_ids.data(), this->num_triangles, 4, 0, 0, 0, &triangleIdsTex, (void**)&d_triangleIds);

                cudaMemcpyToSymbol(textures, &trianglesTex, sizeof(cudaTextureObject_t), 6 * sizeof(cudaTextureObject_t));
                cudaMemcpyToSymbol(textures, &triangleIdsTex, sizeof(cudaTextureObject_t), 7 * sizeof(cudaTextureObject_t));
            }
        }

        streamlines_cuda_impl::~streamlines_cuda_impl()
        {
            if (this->d_velocity) {
                cudaDestroyTextureObject(this->velocity_texture);
                cudaFreeArray(this->d_velocity);
            }

            if (this->d_rk4_step) {
                cudaDestroyTextureObject(this->rk4_step_texture);
                cudaFreeArray(this->d_rk4_step);
            }

            if (this->d_critical_points) {
                cudaDestroyTextureObject(this->critical_points_texture);
                cudaFree(this->d_critical_points);
                cudaDestroyTextureObject(this->critical_point_ids_texture);
                cudaFree(this->d_critical_point_ids);
            }

            if (this->d_segments) {
                cudaDestroyTextureObject(this->segments_texture);
                cudaFree(this->d_segments);
                cudaDestroyTextureObject(this->segment_ids_texture);
                cudaFree(this->d_segment_ids);
            }

            if (this->d_triangles) {
                cudaDestroyTextureObject(this->trianglesTex);
                cudaFree(this->d_triangles);
                cudaDestroyTextureObject(this->triangleIdsTex);
                cudaFree(this->d_triangleIds);
            }
        }

        void streamlines_cuda_impl::update_labels(const std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
            std::vector<float>& end_positions, const int num_integration_steps, const float sign
#if __streamlines_cuda_detailed_output
            , std::vector<float>& integration_steps
#endif
        )
        {
#if __streamlines_cuda_detailed_output
            if (this->resolution[2] != 1)
            {
                throw std::runtime_error("Integration step field only supported for 2D vector fields.");
            }
#endif

            // Subdivide the input
            const int num_particles = static_cast<int>(source->GetNumberOfPoints());
#if __streamlines_cuda_detailed_output
            const int num_particles_per_batch = 2048;
#else
            const int num_particles_per_batch = 20000;
#endif

            cudaError_t err;
            std::stringstream ss;

            const std::size_t max_num_particles_per_batch = std::min(num_particles_per_batch, num_particles);

            // Allocate memory on CPU
            std::vector<short> h_labels(max_num_particles_per_batch);
            std::vector<float> h_dists(max_num_particles_per_batch);
            std::vector<short> h_terminations(max_num_particles_per_batch);
            std::vector<float4> h_particles(max_num_particles_per_batch);

#if __streamlines_cuda_detailed_output
            std::vector<float2> h_integration_steps(this->resolution[0] * this->resolution[1] * max_num_particles_per_batch);
#endif

            // Allocate memory
            short *d_labels;
            err = cudaMalloc((void**)&d_labels, max_num_particles_per_batch * sizeof(short));
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

            short *d_terminations;
            err = cudaMalloc((void**)&d_terminations, max_num_particles_per_batch * sizeof(short));
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            float4 *d_particles;
            err = cudaMalloc((void**)&d_particles, max_num_particles_per_batch * sizeof(float4));
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

            for (int offset = 0; offset < num_particles; offset += num_particles_per_batch)
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

                for (int i = 0; i < num_particles_this_batch; ++i)
                {
                    h_labels[i] = labels_and_distances->GetComponent(i + offset, 0);
                    h_dists[i] = labels_and_distances->GetComponent(i + offset, 1);
                    h_terminations[i] = labels_and_distances->GetComponent(i + offset, 2);

                    std::array<double, 3> p;
                    source->GetPoint(i + offset, p.data());
                    h_particles[i] = make_real<float, 4>(p[0], p[1], p[2], 1.0);
                }

                err = cudaMemcpy(d_labels, h_labels.data(), num_particles_this_batch * sizeof(short), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for labels." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_dists, h_dists.data(), num_particles_this_batch * sizeof(float), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for distances." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_terminations, h_terminations.data(), num_particles_this_batch * sizeof(short), cudaMemcpyHostToDevice);
                if (err)
                {
                    ss << "Error copying to GPU memory using cudaMemcpy for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(d_particles, h_particles.data(), num_particles_this_batch * sizeof(float4), cudaMemcpyHostToDevice);
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
                const int num_steps_per_run = 1000;

                for (int num_steps_completed = 0; num_steps_completed <= num_steps; num_steps_completed += num_steps_per_run)
                {
                    const int num_steps_this_run = std::min(num_steps_per_run, num_steps - num_steps_completed);

                    compute_streamlines(d_particles, num_particles_this_batch, this->num_critical_points, this->num_line_segments,
                        this->num_triangles, num_steps_this_run, sign, d_labels, d_dists, d_terminations
#if __streamlines_cuda_detailed_output    
                        , integration_steps
#endif
                    );
                }
                //--------------------------------------------------------------------------

                // Copy data from GPU memory
                err = cudaMemcpy(h_labels.data(), d_labels, num_particles_this_batch * sizeof(short), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for labels." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(h_dists.data(), d_dists, num_particles_this_batch * sizeof(float), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for distances." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(h_terminations.data(), d_terminations, num_particles_this_batch * sizeof(short), cudaMemcpyDeviceToHost);
                if (err)
                {
                    ss << "Error copying from GPU memory using cudaMemcpy for termination reasons." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                    throw std::runtime_error(ss.str());
                }

                err = cudaMemcpy(h_particles.data(), d_particles, num_particles_this_batch * sizeof(float4), cudaMemcpyDeviceToHost);
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

                // Store information in output variables
                for (int i = 0; i < num_particles_this_batch; ++i)
                {
                    labels_and_distances->SetComponent(i + offset, 0, h_labels[i]);
                    labels_and_distances->SetComponent(i + offset, 1, h_dists[i]);
                    labels_and_distances->SetComponent(i + offset, 2, h_terminations[i]);

                    end_positions->SetComponent(i + offset, 0, h_particles[i].x);
                    end_positions->SetComponent(i + offset, 1, h_particles[i].y);
                    end_positions->SetComponent(i + offset, 2, h_particles[i].z);

                    source->GetPoints()->SetPoint(i + offset, h_particles[i].x, h_particles[i].y, h_particles[i].z);

#if __streamlines_cuda_detailed_output
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
#endif
                }
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

        void streamlines_cuda_impl::compute_streamlines(float4* d_particles, const int num_particles, const int num_critical_points, const int num_segments,
            const int num_triangles, const int num_steps, const float sign, short* d_labels, float* d_dists, short* d_terminations
#if __streamlines_cuda_detailed_output
            , cudaSurfaceObject_t integration_steps
#endif
        )
        {
            // Run CUDA kernel
            int num_threads = 64;
            int num_blocks = num_particles / num_threads + (num_particles % num_threads == 0 ? 0 : 1);

            compute_streamlines_kernel << <num_blocks, num_threads >> > (num_critical_points, num_segments, num_triangles, sign,
                d_particles, num_particles, num_steps, d_labels, d_dists, d_terminations
#if __streamlines_cuda_detailed_output
                , integration_steps
#endif
                );
        }

        void streamlines_cuda_impl::initialize_texture(void* h_data, const int num_components, cudaTextureObject_t* texture, cudaArray** d_data)
        {
            const cudaExtent extent = { static_cast<std::size_t>(this->resolution[0]), static_cast<std::size_t>(this->resolution[1]), static_cast<std::size_t>(this->resolution[2]) };
            cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, num_components > 1 ? 32 : 0,
                num_components > 2 ? 32 : 0, num_components > 3 ? 32 : 0, cudaChannelFormatKindFloat);

            cudaError_t err;
            std::stringstream ss;

            err = cudaMalloc3DArray(d_data, &desc, extent);
            if (err)
            {
                ss << "Error allocating memory using cudaMalloc3DArray for velocity or RK4 step size." << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
                throw std::runtime_error(ss.str());
            }

            cudaPitchedPtr srcPtr;
            srcPtr.ptr = h_data;
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

        void streamlines_cuda_impl::initialize_texture(void* h_data, const int num_elements, const int c0, const int c1, const int c2, const int c3, cudaTextureObject_t* texture, void** d_data)
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