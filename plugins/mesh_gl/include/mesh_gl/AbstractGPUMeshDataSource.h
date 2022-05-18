/*
 * AbstractGPUMeshDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "GPUMeshCollection.h"

namespace megamol {
namespace mesh_gl {
class AbstractGPUMeshDataSource : public core::Module {
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
     * Gets the meta data from the source.
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
     * Clears all render task entries made by this module from the used material collection.
     */
    void clearMeshCollection();

    /**
     * Mesh collection that is used with a list of identifier strings of all GPU submeshes that this module added to
     * the mesh collection. Needed to delete/update submeshes if the collection is shared across a chain of data
     * sources modules.
     */
    std::pair<std::shared_ptr<GPUMeshCollection>, std::vector<std::string>> m_mesh_collection;

    /** The slot for querying additional mesh data, i.e. a rhs chaining connection */
    megamol::core::CallerSlot m_mesh_rhs_slot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_mesh_lhs_slot;
};

} // namespace mesh_gl
} // namespace megamol


#endif // !ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
