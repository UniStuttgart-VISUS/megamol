/*
 * FEMGPUMeshDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef FEM_GPU_MESH_DATA_SOURCE_H_INCLUDED
#define FEM_GPU_MESH_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"

#include "mesh/AbstractGPUMeshDataSource.h"


namespace megamol {
namespace archvis {

class FEMMeshDataSource : public mesh::AbstractGPUMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FEMMeshDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for generating and uploading mesh data from FEM data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }


    FEMMeshDataSource();
    ~FEMMeshDataSource();

protected:

    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    megamol::core::CallerSlot m_fem_callerSlot;

    uint64_t m_FEM_model_hash;
};

} // namespace archvis
} // namespace megamol

#endif // !FEM_GPU_MESH_DATA_SOURCE_H_INCLUDED
