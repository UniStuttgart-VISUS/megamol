/*
 * Utility.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MESH_UTILITY_H_INCLUDED
#define MESH_UTILITY_H_INCLUDED

#include <array>
#include <tuple>
#include <vector>

#include "glm/glm.hpp"

namespace megamol::mesh::utility {

using VertexPositions = std::vector<std::array<float, 3>>;
using VertexNormals = std::vector<std::array<float, 3>>;
using QuadIndices = std::vector<std::array<uint32_t, 4>>;

std::tuple<VertexPositions, VertexNormals, QuadIndices> tessellateFace(
    glm::vec3 v00, glm::vec3 v10, glm::vec3 v11, glm::vec3 v01, unsigned int u_subdivs, unsigned int v_subdivs);

} // namespace megamol::mesh::utility

#endif // !MESH_UTILITY_H_INCLUDED
