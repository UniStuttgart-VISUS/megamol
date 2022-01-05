/*
 * TessellateBoundingBox.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef TESSELLATE_BOUNDING_BOX_H_INCLUDED
#define TESSELLATE_BOUNDING_BOX_H_INCLUDED

#include "mesh/AbstractMeshDataSource.h"
#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace probe {

class TessellateBoundingBox : public mesh::AbstractMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TessellateBoundingBox";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Tessellate a 3D Bounding Box into actual surface geometry";
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
    TessellateBoundingBox(void);

    /** Dtor. */
    virtual ~TessellateBoundingBox(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _bounding_box_rhs_slot;

    core::param::ParamSlot _x_subdiv_slot;
    core::param::ParamSlot _y_subdiv_slot;
    core::param::ParamSlot _z_subdiv_slot;

    core::param::ParamSlot _padding_slot;

private:
    bool InterfaceIsDirty();

    bool getMeshDataCallback(core::Call& caller);

    bool getMeshMetaDataCallback(core::Call& caller);

    uint32_t _version = 0;
    size_t _old_datahash;
    bool _recalc = false;

    // store surfaces of the six tessellated sides of the bounding box
    std::array<std::vector<std::array<float, 3>>, 6> _vertices;
    std::array<std::vector<std::array<float, 3>>, 6> _normals;
    std::array<std::vector<int>, 6> _probe_index;
    std::array<std::vector<std::array<uint32_t, 4>>, 6> _faces;

    // store bounding box
    megamol::core::BoundingBoxes_2 _bboxs;
};

} // namespace probe
} // namespace megamol

#endif // !TESSELLATE_BOUNDING_BOX_H_INCLUDED
