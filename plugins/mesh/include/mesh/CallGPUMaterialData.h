/*
 * CallGPUMaterialData.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MATERIAL_CALL_H_INCLUDED
#define GPU_MATERIAL_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPUMaterialCollection.h"
#include "mmcore/AbstractGetDataCall.h"
#include "mesh.h"

namespace megamol {
namespace mesh {
class MESH_API CallGPUMaterialData : public megamol::core::AbstractGetDataCall {
public:
    inline CallGPUMaterialData() : AbstractGetDataCall(), m_gpu_materials(nullptr) {}
    ~CallGPUMaterialData() = default;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallGPUMaterialData"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call that gives access to material data stored on the GPU."; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetDataCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetDataCall::FunctionName(idx); }

    void setMaterialStorage(std::shared_ptr<GPUMaterialCollecton> const& gpu_materials) {
        m_gpu_materials = gpu_materials;
    }

    std::shared_ptr<GPUMaterialCollecton> getMaterialStorage() { return m_gpu_materials; }

private:
    std::shared_ptr<GPUMaterialCollecton> m_gpu_materials;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallGPUMaterialData> CallGPUMaterialDataDescription;

} // namespace mesh
} // namespace megamol

#endif // !GPU_MATERIAL_CALL_H_INCLUDED
