/*
 * ManipulateMesh.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/MeshUtilities.h"

namespace megamol {
namespace probe {

class ManipulateMesh : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ManipulateMesh";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Manipulates Mesh data using libigl";
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
    ManipulateMesh(void);

    /** Dtor. */
    virtual ~ManipulateMesh(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataSlot;
    core::CalleeSlot _deployMeshSlot;
    core::CalleeSlot _deployNormalsSlot;
    core::CalleeSlot _pointsDebugSlot;

    core::param::ParamSlot _numFacesSlot;

private:
    bool InterfaceIsDirty();

    bool performMeshOperation(const mesh::MeshDataAccessCollection::Mesh&);
    bool convertToMesh();

    bool getMetaData(core::Call& call);
    bool getParticleMetaData(core::Call& call);
    bool getParticleData(core::Call& call);
    bool getData(core::Call& call);

    bool parameterChanged(core::param::ParamSlot& p);

    // CallMesh stuff
    std::vector<float> _mesh_vertices;
    std::vector<uint32_t> _mesh_faces;
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    uint32_t _version = 0;

    bool _recalc = true;

    mesh::MeshDataAccessCollection::VertexAttribute _pos_attribute;

    std::shared_ptr<MeshUtility> _mu;
    core::BoundingBoxes_2 _bbox;
    std::vector<float> _points;
    std::vector<Eigen::MatrixXd> _pointsPerFace;
    std::map<uint32_t, std::vector<uint32_t>> _neighborMap;
    uint32_t _numFaces = 0;
};

} // namespace probe
} // namespace megamol
