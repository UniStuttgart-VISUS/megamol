/*
 * AbstractGPURenderTaskDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPURenderTaskCollection.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mesh.h"

namespace megamol {
namespace mesh {

class MESH_API AbstractGPURenderTaskDataSource : public core::Module {
public:
    AbstractGPURenderTaskDataSource();
    virtual ~AbstractGPURenderTaskDataSource();

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
     * This modules storage class for Render Tasks.
     * If connected within a chain of rts (but not the first rt module), the storage should remain unused
     * and instead the collection provided by the left-hand-side rt is used.
     */
    std::shared_ptr<GPURenderTaskCollection> m_gpu_render_tasks;

    /**
     * List of indices of all RenderTasks that this module added to the used rt collection.
     * Needed to delete/update RenderTasks if the rt collection is shared across a chain of rt data sources.
     */
    std::vector<size_t> m_rt_collection_indices;

    /** The slot for requesting data from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_getData_slot;

    /** The slot for querying chained render tasks, i.e. a rhs connection */
    megamol::core::CallerSlot m_renderTask_callerSlot;

    /** The slot for querying material data, i.e. a rhs connection */
    megamol::core::CallerSlot m_material_callerSlot;

    /** The slot for querying mesh data, i.e. a rhs connection */
    megamol::core::CallerSlot m_mesh_callerSlot;
};

} // namespace mesh
} // namespace megamol

#endif // !ABSTRACT_RENDER_TASK_DATA_SOURCE_H_INCLUDED
