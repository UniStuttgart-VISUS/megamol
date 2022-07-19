/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/AbstractGPUMeshDataSource.h"
#include "mmcore/CallerSlot.h"

namespace megamol::archvis_gl {

class FEMMeshDataSource : public mesh_gl::AbstractGPUMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "FEMMeshDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source for generating and uploading mesh data from FEM data";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }


    FEMMeshDataSource();
    ~FEMMeshDataSource();

protected:
    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    megamol::core::CallerSlot m_fem_callerSlot;
};

} // namespace megamol::archvis_gl
