/*
 * SimpleMeshViewerRenderTasks.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_VIEWER_RENDER_TASK_H_INCLUDED
#define MESH_VIEWER_RENDER_TASK_H_INCLUDED

#include "mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol {
namespace mesh {

class MeshViewerRenderTasks : public AbstractGPURenderTaskDataSource
{
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "MeshViewerRenderTasks"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Simple mesh viewer: Creates a single render task for each available GPU mesh."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    MeshViewerRenderTasks();
    ~MeshViewerRenderTasks();

protected:

    virtual bool getDataCallback(core::Call& caller);
};

}
}


#endif // !SIMPLE_MESH_VIEWER_RENDER_TASK_H_INCLUDED
