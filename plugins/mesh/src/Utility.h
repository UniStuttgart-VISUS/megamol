/*
 * Utility.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MESH_UTILITY_H_INCLUDED
#define MESH_UTILITY_H_INCLUDED

#include <array>
#include <vector>
#include <tuple>

#include "glm/glm.hpp"

namespace megamol {
namespace mesh {
namespace utility {

    using VertexPositions = std::vector<std::array<float, 3>>;
    using VertexNormals = std::vector<std::array<float, 3>>;
    using QuadFaceIndices = std::vector<std::array<uint32_t, 4>>;

    std::tuple<VertexPositions, VertexNormals, QuadFaceIndices> tessellateFace(
        glm::vec3 v00, glm::vec3 v10, glm::vec3 v11, glm::vec3 v01, unsigned int u_subdivs, unsigned int v_subdivs);

} // namespace utility
} // namespace mesh
} // namespace megamol

#endif // !MESH_UTILITY_H_INCLUDED
