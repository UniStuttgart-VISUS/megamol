/*
 * MSMConvexHullMeshDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MSM_CONVEX_HULL_MESH_DATA_SOURCE_H_INCLUDED
#define MSM_CONVEX_HULL_MESH_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"

#include "mesh/AbstractGPUMeshDataSource.h"

namespace megamol {
namespace archvis {

class MSMConvexHullDataSource : public mesh::AbstractGPUMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "MSMConvexHullDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for generating convex hulls from MSM displacement values."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    MSMConvexHullDataSource();
    ~MSMConvexHullDataSource();

protected:

    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    megamol::core::CallerSlot m_MSM_callerSlot;

    uint64_t m_MSM_hash;
};

}
}


#endif // !MSM_CONVEX_HULL_MESH_DATA_SOURCE_H_INCLUDED
