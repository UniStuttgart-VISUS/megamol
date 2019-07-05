/*
 * CallGPUMeshData.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MESH_DATA_CALL_H_INCLUDED
#define GPU_MESH_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPUMeshCollection.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mesh.h"

namespace megamol {
namespace mesh {

class CallGPUMeshData : public megamol::core::AbstractGetData3DCall {
public:
    CallGPUMeshData() : AbstractGetData3DCall(), m_gpu_meshes(nullptr) {}
    ~CallGPUMeshData() = default;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallGPUMeshData"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call that gives access to meshes stored in batches on the GPU for rendering.";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetData3DCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetData3DCall::FunctionName(idx); }

    void setGPUMeshes(std::shared_ptr<GPUMeshCollection> gpu_meshes) { m_gpu_meshes = gpu_meshes; }

    std::shared_ptr<GPUMeshCollection> getGPUMeshes() { return m_gpu_meshes; }

private:
    std::shared_ptr<GPUMeshCollection> m_gpu_meshes;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallGPUMeshData> CallGPUMeshDataDescription;

} // namespace mesh
} // namespace megamol


#endif // !GPU_MESH_DATA_CALL_H_INCLUDED
