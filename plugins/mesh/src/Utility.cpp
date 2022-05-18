#include "mesh/Utility.h"

namespace megamol {
namespace mesh {
namespace utility {

std::tuple<VertexPositions, VertexNormals, QuadIndices> tessellateFace(
    glm::vec3 v00, glm::vec3 v10, glm::vec3 v11, glm::vec3 v01, unsigned int u_subdivs, unsigned int v_subdivs) {

    VertexPositions vertex_positions;
    VertexNormals vertex_normals;
    QuadIndices quad_indices;

    auto normal = -1.0f * glm::cross(glm::normalize(v10 - v00), glm::normalize(v01 - v00));

    // compute vertex positions using bilinear interpolation
    for (int v = 0; v <= v_subdivs + 1; ++v) {

        auto lambda_v = static_cast<float>(v) / static_cast<float>(v_subdivs + 1);
        auto p_v_0 = lambda_v * v00 + (1.0f - lambda_v) * v01;
        auto p_v_1 = lambda_v * v10 + (1.0f - lambda_v) * v11;

        for (int u = 0; u <= u_subdivs + 1; ++u) {

            auto lambda_u = static_cast<float>(u) / static_cast<float>(u_subdivs + 1);
            auto p = lambda_u * p_v_0 + (1.0f - lambda_u) * p_v_1;
            vertex_positions.push_back({p.x, p.y, p.z});
            vertex_normals.push_back({normal.x, normal.y, normal.z});
        }
    }

    // iterate subdivs again and create faces
    for (int v = 0; v < v_subdivs + 1; ++v) {
        for (int u = 0; u < u_subdivs + 1; ++u) {
            // # faces per row = # u_subdivs + vertices on edges
            uint32_t base_idx = v * (u_subdivs + 2) + u;

            quad_indices.push_back(
                {base_idx, base_idx + 1, base_idx + (u_subdivs + 2) + 1, base_idx + (u_subdivs + 2)});
        }
    }

    return std::make_tuple<VertexPositions, VertexNormals, QuadIndices>(
        std::move(vertex_positions), std::move(vertex_normals), std::move(quad_indices));
}

} // namespace utility
} // namespace mesh
} // namespace megamol
