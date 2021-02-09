/*
 * AbstractGPUMaterialDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPUMaterialCollection.h"
#include "mmcore/CalleeSlot.h"
#include "mesh/MeshCalls.h"

namespace megamol {
namespace mesh {

class AbstractGPUMaterialDataSource : public core::Module {
public:
    AbstractGPUMaterialDataSource();
    virtual ~AbstractGPUMaterialDataSource();

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

    virtual bool getMetaDataCallback(core::Call& caller) = 0;

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    void syncMaterialCollection(CallGPUMaterialData* lhs_call);

    /**
     * Material collection that is used and indices of materials within the collection that were added by a module instance.
     */
    std::pair<std::shared_ptr<GPUMaterialCollection>, std::vector<std::string>> m_material_collection;

    /** The slot for querying additional material data, i.e. a rhs chaining connection */
    megamol::core::CallerSlot m_mtl_callerSlot;

private:
    /** The slot for requesting data from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_getData_slot;
};

} // namespace mesh
} // namespace megamol

#endif // !ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED