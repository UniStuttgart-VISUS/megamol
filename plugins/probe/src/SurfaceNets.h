/*
 * SurfaceNets.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "concave_hull.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "poisson.h"
#include <cstdlib>

namespace megamol {
namespace probe {

class SurfaceNets : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SurfaceNets";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Extracts a surface mesh from volume data.";
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
    SurfaceNets(void);

    /** Dtor. */
    virtual ~SurfaceNets(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployMeshCall;
    core::CalleeSlot _deployNormalsCall;

    core::param::ParamSlot _isoSlot;
    core::param::ParamSlot _faceTypeSlot;


private:
    bool InterfaceIsDirty();

    void calculateSurfaceNets();

    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool isoChanged(core::param::ParamSlot& p);

    bool getNormalData(core::Call& call);

    bool getNormalMetaData(core::Call& call);

    // CallMesh stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    mesh::MeshDataAccessCollection::PrimitiveType _mesh_type;
    uint32_t _version = 0;

    size_t _old_datahash;
    bool _recalc = false;

    std::array<uint32_t, 3> _dims;
    std::array<float, 3> _spacing;
    std::array<float, 3> _volume_origin;
    std::vector<float> _converted_data;
    float* _data;

    // store surface
    std::vector<std::array<float, 4>> _vertices;
    std::vector<std::array<float, 3>> _normals;
    std::vector<std::array<uint32_t, 4>> _faces;
    std::vector<std::array<uint32_t, 3>> _triangles;

    // store bounding box
    megamol::core::BoundingBoxes_2 _bboxs;
};

} // namespace probe
} // namespace megamol
