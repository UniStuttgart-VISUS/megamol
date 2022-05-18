/*
 * MeshCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mesh/MeshCalls.h"

#include "GPUMaterialCollection.h"
#include "GPUMeshCollection.h"
#include "GPURenderTaskCollection.h"

namespace megamol {
namespace mesh_gl {

class CallGPUMaterialData
        : public core::GenericVersionedCall<std::vector<std::shared_ptr<GPUMaterialCollection>>, core::EmptyMetaData> {
public:
    CallGPUMaterialData()
            : GenericVersionedCall<std::vector<std::shared_ptr<GPUMaterialCollection>>, core::EmptyMetaData>() {}
    ~CallGPUMaterialData() = default;

    static const char* ClassName(void) {
        return "CallGPUMaterialData";
    }
    static const char* Description(void) {
        return "Call that gives access to material data stored on the GPU.";
    }
};

class CallGPUMeshData
        : public core::GenericVersionedCall<std::vector<std::shared_ptr<GPUMeshCollection>>, core::Spatial3DMetaData> {
public:
    CallGPUMeshData()
            : GenericVersionedCall<std::vector<std::shared_ptr<GPUMeshCollection>>, core::Spatial3DMetaData>() {}
    ~CallGPUMeshData() = default;

    static const char* ClassName(void) {
        return "CallGPUMeshData";
    }
    static const char* Description(void) {
        return "Call that gives access to meshes stored in batches on the GPU for rendering.";
    }
};

class CallGPURenderTaskData : public core::GenericVersionedCall<std::vector<std::shared_ptr<GPURenderTaskCollection>>,
                                  core::Spatial3DMetaData> {
public:
    CallGPURenderTaskData()
            : GenericVersionedCall<std::vector<std::shared_ptr<GPURenderTaskCollection>>, core::Spatial3DMetaData>() {}
    ~CallGPURenderTaskData(){};

    static const char* ClassName(void) {
        return "CallGPURenderTaskData";
    }
    static const char* Description(void) {
        return "Call that gives access to render tasks.";
    }
};


/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallGPURenderTaskData> GPURenderTasksDataCallDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMeshData> CallGPUMeshDataDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMaterialData> CallGPUMaterialDataDescription;

} // namespace mesh_gl
} // namespace megamol
