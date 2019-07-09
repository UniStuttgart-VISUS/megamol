/*
 * glTFMeshesDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GLTF_MESHES_DATA_SOURCE_H_INCLUDED
#define GLTF_MESHES_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"

#include "mesh/AbstractGPUMeshDataSource.h"

namespace megamol {
namespace mesh {

class GlTFMeshesDataSource : public AbstractGPUMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GlTFMeshesDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for uploading mesh data from a glTF file"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }


    GlTFMeshesDataSource();
    ~GlTFMeshesDataSource();

protected:
    virtual bool create();

    virtual bool getDataCallback(core::Call& caller);

private:
    megamol::core::CallerSlot m_glTF_callerSlot;
};

} // namespace mesh
} // namespace megamol

#endif // !GLTF_MESHES_DATA_SOURCE_H_INCLUDED
