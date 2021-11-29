/*
 * AbstractGPURenderTaskDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPURenderTaskCollection.h"
#include "mesh_gl/MeshCalls_gl.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/light/CallLight.h"

namespace megamol {
namespace mesh_gl {

class AbstractGPURenderTaskDataSource : public core::Module {
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
    virtual bool getMetaDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    /**
     * Clears all render tasks
     */
    void clearRenderTaskCollection();

    /**
     * Render task collection that is used with a list of indices of all RenderTasks that this module added to the
     * used rt collection. Needed to delete/update RenderTasks if the rt collection is shared across a chain of rt
     * data sources.
     */
    std::pair<std::shared_ptr<GPURenderTaskCollection>, std::vector<std::string>> m_rendertask_collection;

    /** The slot for requesting data from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_renderTask_lhs_slot;

    /** The slot for querying chained render tasks, i.e. a rhs connection */
    megamol::core::CallerSlot m_renderTask_rhs_slot;

    /** The slot for querying mesh data, i.e. a rhs connection */
    megamol::core::CallerSlot m_mesh_slot;
};

} // namespace mesh_gl
} // namespace megamol

#endif // !ABSTRACT_RENDER_TASK_DATA_SOURCE_H_INCLUDED
