/*
 * ExtractSkeleton.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <cstdlib>

namespace megamol {
namespace probe {

class ExtractSkeleton : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ExtractSkeleton";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Extracts a skeleton of the incomming mesh data.";
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
    ExtractSkeleton(void);

    /** Dtor. */
    virtual ~ExtractSkeleton(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployLineCall;


private:
    bool InterfaceIsDirty();
    bool compute(float* vertices, uint32_t num_vertices, uint32_t num_components);

    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool usePoisson = true;
    std::vector<float> _vertex_data;
    std::vector<float> _normal_data;

    // CallMesh stuff
    core::BoundingBoxes_2 _bbox;
    uint32_t _version = 0;

    // CallCenterline stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _line_attribs;
    mesh::MeshDataAccessCollection::IndexData _line_indices;
    std::vector<std::array<float, 4>> _centerline;
    std::vector<std::vector<uint32_t>> _cl_indices_per_slice;
    std::vector<uint32_t> _cl_indices;

    size_t _old_datahash = 0;
};

} // namespace probe
} // namespace megamol
