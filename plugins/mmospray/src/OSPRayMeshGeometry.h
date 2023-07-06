/*
 * OSPRayMeshGeometry.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol::ospray {

class OSPRayMeshGeometry : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayMeshGeometry";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creator for OSPRay mesh geometries.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Dtor. */
    ~OSPRayMeshGeometry() override;

    /** Ctor. */
    OSPRayMeshGeometry();

protected:
    bool create() override;
    void release() override;

    bool readData(core::Call& call) override;
    bool getExtends(core::Call& call) override;


    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot _getMeshDataSlot;
    std::vector<float> _color;
    std::vector<size_t> _mesh_prefix_count;
};

} // namespace megamol::ospray
