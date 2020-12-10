#include "Utility.h"

namespace megamol {
namespace mesh {
namespace utility {

std::tuple<VertexPositions, std::vector<std::array<float, 3>>, QuadFaceIndices> tessellateFace(
    glm::vec3 v00, glm::vec3 v10, glm::vec3 v11, glm::vec3 v01, unsigned int u_subdivs, unsigned int v_subdivs) {

    VertexPositions vertex_positions;
    VertexNormals vertex_normals;
    QuadFaceIndices quad_indices;

    // TODO compute vector for u step and v step


    // TODO iterate subdivs, create faces as you go


    return std::make_tuple<VertexPositions, VertexNormals, QuadFaceIndices>(
        std::move(vertex_positions), std::move(vertex_normals), std::move(quad_indices));
}

} // namespace utility
} // namespace mesh
} // namespace megamol
