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
    virtual bool getExtentCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    /**
     * The bounding box stored as left,bottom,back,right,top,front
     */
    std::array<float, 6> m_bbox;


    std::shared_ptr<GPUMeshCollection> m_gpu_meshes;

    /** The slot for querying additional mesh data, i.e. a rhs chaining connection */
    megamol::core::CallerSlot m_mesh_callerSlot;

private:
    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !ABSTRACT_GPU_MESH_DATA_SOURCE_H_INCLUDED
