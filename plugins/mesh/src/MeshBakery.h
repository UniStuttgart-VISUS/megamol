/*
 * MeshBakery.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_BAKERY_H_INCLUDED
#    define MESH_BAKERY_H_INCLUDED

#    include "mmcore/CalleeSlot.h"
#    include "mmcore/param/ParamSlot.h"

#    include "mesh/MeshCalls.h"
#    include "mesh/MeshDataAccessCollection.h"
#    include "mesh/mesh.h"

namespace megamol {
namespace mesh {


class MeshBakery : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "MeshBakery"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    MeshBakery();
    ~MeshBakery();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:

    void createTriangleGeometry();

    void createConeGeometry();

    uint32_t m_version;

    std::shared_ptr<MeshDataAccessCollection> m_mesh_access_collection;

    std::vector<float>   m_vertex_positions;
    std::vector<float>   m_vertex_normals;
    std::vector<float>   m_vertex_tangents;
    std::vector<float>   m_vertex_uvs;
    std::vector<uint8_t> m_vertex_colors;

    std::vector<uint32_t> m_indices;


    /** Parameter for selecting the geometry to be generated */
    megamol::core::param::ParamSlot m_geometry_type;

    /** The slot for providing access to internal mesh data */
    megamol::core::CalleeSlot m_mesh_lhs_slot;

    /** The slot for chaining mesh data access */
    megamol::core::CallerSlot m_mesh_rhs_slot;

};

} // namespace mesh
} // namespace megamol

#endif // !MESH_BAKERY_H_INCLUDED
