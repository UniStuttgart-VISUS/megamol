/*
 * PlaceProbes.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PLACE_PROBES_H_INCLUDED
#define PLACE_PROBES_H_INCLUDED

#include "mesh/MeshCalls.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/MeshUtilities.h"
#include "probe/ProbeCollection.h"

namespace megamol {
namespace probe {

class PlaceProbes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "PlaceProbes";
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
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    PlaceProbes();

    /** Dtor. */
    virtual ~PlaceProbes();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::CallerSlot _mesh_slot;

    core::CallerSlot _centerline_slot;

    core::CallerSlot _load_probe_positions_slot;

    core::CalleeSlot _probe_slot;

    core::CalleeSlot _probe_positions_slot;

    core::param::ParamSlot _method_slot;
    core::param::ParamSlot _probes_per_unit_slot;
    core::param::ParamSlot _scale_probe_begin_slot;


private:
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    void dartSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        mesh::MeshDataAccessCollection::IndexData indexData, float distanceIndicator);
    void forceDirectedSampling(const mesh::MeshDataAccessCollection::Mesh& mesh);
    void vertexSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices);
    void vertexNormalSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
        mesh::MeshDataAccessCollection::VertexAttribute& normals,
        mesh::MeshDataAccessCollection::VertexAttribute& probe_ids);
    void faceNormalSampling(mesh::MeshDataAccessCollection::VertexAttribute const& vertices,
        mesh::MeshDataAccessCollection::VertexAttribute const& normals,
        mesh::MeshDataAccessCollection::VertexAttribute const& probe_ids,
        mesh::MeshDataAccessCollection::IndexData const& indices);
    bool placeProbes();
    bool placeByCenterline(uint32_t lei, mesh::MeshDataAccessCollection::VertexAttribute& centerline);
    bool placeByCenterpoint();
    bool getADIOSData(core::Call& call);
    bool getADIOSMetaData(core::Call& call);
    bool loadFromFile();
    bool parameterChanged(core::param::ParamSlot& p);

    uint32_t _longest_edge_index;

    std::shared_ptr<ProbeCollection> _probes;
    std::shared_ptr<mesh::MeshDataAccessCollection> _mesh;
    std::shared_ptr<mesh::MeshDataAccessCollection> _centerline;
    std::array<float, 3> _whd;
    core::BoundingBoxes_2 _bbox;

    // force directed stuff
    std::shared_ptr<MeshUtility> _mu;
    std::vector<Eigen::MatrixXd> _pointsPerFace;
    std::map<uint32_t, std::vector<uint32_t>> _neighborMap;
    uint32_t _numFaces = 0;
    std::vector<std::array<float, 4>> _probePositions;
    std::vector<uint64_t> _probeVertices;
    adios::adiosDataMap dataMap;
    bool _recalc;
};


} // namespace probe
} // namespace megamol

#endif //!PLACE_PROBES_H_INCLUDED
