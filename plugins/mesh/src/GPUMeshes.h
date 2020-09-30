/*
 * GPUMeshes.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MESHES_H_INCLUDED
#define GPU_MESHES_H_INCLUDED

#include "mesh/AbstractGPUMeshDataSource.h"


namespace megamol {
namespace mesh {


class GPUMeshes : public AbstractGPUMeshDataSource
{
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GPUMeshes"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for uploading mesh data from a CPU-side mesh call"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    GPUMeshes();
    ~GPUMeshes();

protected:

    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    megamol::core::CallerSlot m_mesh_slot;
};


}
}


#endif // !GPU_MESHES_H_INCLUDED