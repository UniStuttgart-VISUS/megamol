/*
 * AbstractGPUMeshDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mesh.h"

#include "GPUMeshCollection.h"

namespace megamol {
namespace mesh {
class MESH_API AbstractGPUMeshDataSource : public core::Module {
public:
    AbstractGPUMeshDataSource();
    virtual ~AbstractGPUMeshDataSource();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getDataCallback(core::Call& caller) = 0;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getMetaDataCallback(core::Call& caller) = 0;

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    /**
     * This module's storage class for GPU meshes.
     * If connected within a chain of mesh datasource (but not the first module in the chain),
     * this storage should remain unused and instead the collection provided by the left-hand-side module is used.
     */
    std::shared_ptr<GPUMeshCollection> m_gpu_meshes;

    /**
     * List of indices of all GPU submeshes that this module added to the used mesh collection.
     * Needed to delete/update submeshes if the collection is shared across a chain of data sources modules.
     */
    std::vector<size_t> m_mesh_collection_indices;

    /** The slot for querying additional mesh data, i.e. a rhs chaining connection */
    megamol::core::CallerSlot m_mesh_rhs_slot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_mesh_lhs_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
