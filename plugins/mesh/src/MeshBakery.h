/*
 * MeshBakery.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/AbstractMeshDataSource.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataAccessCollection.h"

namespace megamol::mesh {


class MeshBakery : public AbstractMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MeshBakery";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    MeshBakery();
    ~MeshBakery() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    bool getMeshDataCallback(core::Call& caller) override;

    bool getMeshMetaDataCallback(core::Call& caller) override;

private:
    void createTriangleGeometry();

    void createConeGeometry();

    uint32_t m_version;

    std::vector<float> m_vertex_positions;
    std::vector<float> m_vertex_normals;
    std::vector<float> m_vertex_tangents;
    std::vector<float> m_vertex_uvs;
    std::vector<uint8_t> m_vertex_colors;

    std::vector<uint32_t> m_indices;

    /** Parameter for selecting the geometry to be generated */
    megamol::core::param::ParamSlot m_geometry_type;
};

} // namespace megamol::mesh
