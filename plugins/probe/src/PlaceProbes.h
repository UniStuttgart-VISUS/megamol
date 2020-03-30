/*
 * PlaceProbes.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PLACE_PROBES_H_INCLUDED
#define PLACE_PROBES_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mesh/MeshCalls.h"
#include "ProbeCollection.h"

namespace megamol {
namespace probe {

class PlaceProbes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "PlaceProbes"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    PlaceProbes();

    /** Dtor. */
    virtual ~PlaceProbes();

protected:
    virtual bool create();
    virtual void release();

    uint32_t m_version;

    core::CallerSlot m_mesh_slot;

    core::CallerSlot m_centerline_slot;

    core::CalleeSlot m_probe_slot;

    core::param::ParamSlot m_method_slot;

    core::param::ParamSlot m_probes_per_unit_slot;
    
private:
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    void dartSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        std::vector<std::array<float, 4>>& output, mesh::MeshDataAccessCollection::IndexData indexData,
        float distanceIndicator);
    void forceDirectedSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        std::vector<std::array<float, 4>>& output);
    void vertexSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        std::vector<std::array<float, 4>>& output);
    void vertexNormalSampling(
        mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        mesh::MeshDataAccessCollection::VertexAttribute& normals);
    bool placeProbes(uint32_t lei);
    bool placeByCenterline(uint32_t lei, std::vector<std::array<float, 4>>& probePositions,
                           mesh::MeshDataAccessCollection::VertexAttribute& centerline);

    std::shared_ptr<ProbeCollection> m_probes;
    std::shared_ptr<mesh::MeshDataAccessCollection> m_mesh;
    std::shared_ptr<mesh::MeshDataAccessCollection> m_centerline;
    std::array<float, 3> m_whd;

};


} // namespace probe
} // namespace megamol

#endif //!PLACE_PROBES_H_INCLUDED