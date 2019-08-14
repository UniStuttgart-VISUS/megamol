/*
 * MeshCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_CALLS_H_INCLUDED
#define MESH_CALLS_H_INCLUDED

#include "CallGeneric.h"

#include <memory>
#include "tiny_gltf.h"

#include "3DInteractionCollection.h"
#include "GPUMaterialCollection.h"
#include "GPUMeshCollection.h"
#include "GPURenderTaskCollection.h"

namespace megamol {
namespace mesh {

struct BasicMetaData {
    size_t m_data_hash = 0;
};

struct Spatial3DMetaData {
    size_t                       m_data_hash = 0;
    unsigned int                 m_frame_cnt = 0;
    unsigned int                 m_frame_ID = 0;
    megamol::core::BoundingBoxes m_bboxs;
};

class MESH_API Call3DInteraction
    : public CallGeneric<std::shared_ptr<ThreeDimensionalInteractionCollection>, BasicMetaData> {
public:
    inline Call3DInteraction() : CallGeneric<std::shared_ptr<ThreeDimensionalInteractionCollection>, BasicMetaData>() {}
    ~Call3DInteraction() = default;

    static const char* ClassName(void) { return "Call3DInteraction"; }
    static const char* Description(void) { return "Call that transports..."; }
};

class MESH_API CallGlTFData : public CallGeneric<std::shared_ptr<tinygltf::Model>, BasicMetaData> {
public:
    inline CallGlTFData() : CallGeneric<std::shared_ptr<tinygltf::Model>, BasicMetaData>() {}
    ~CallGlTFData() = default;

    static const char* ClassName(void) { return "CallGlTFData"; }
    static const char* Description(void) { return "Call that gives access to a loaded gltf model."; }
};

class MESH_API CallGPUMaterialData : public CallGeneric<std::shared_ptr<GPUMaterialCollecton>, BasicMetaData> {
public:
    inline CallGPUMaterialData() : CallGeneric<std::shared_ptr<GPUMaterialCollecton>, BasicMetaData>() {}
    ~CallGPUMaterialData() = default;

    static const char* ClassName(void) { return "CallGPUMaterialData"; }
    static const char* Description(void) { return "Call that gives access to material data stored on the GPU."; }
};

class MESH_API CallGPUMeshData : public CallGeneric<std::shared_ptr<GPUMeshCollection>, Spatial3DMetaData> {
public:
    CallGPUMeshData() : CallGeneric<std::shared_ptr<GPUMeshCollection>, Spatial3DMetaData>() {}
    ~CallGPUMeshData() = default;

    static const char* ClassName(void) { return "CallGPUMeshData"; }
    static const char* Description(void) {
        return "Call that gives access to meshes stored in batches on the GPU for rendering.";
    }
};

class MESH_API CallGPURenderTaskData : public CallGeneric<std::shared_ptr<GPURenderTaskCollection>, Spatial3DMetaData> {
public:
    inline CallGPURenderTaskData() : CallGeneric<std::shared_ptr<GPURenderTaskCollection>, Spatial3DMetaData>() {}
    ~CallGPURenderTaskData(){};

    static const char* ClassName(void) { return "CallGPURenderTaskData"; }
    static const char* Description(void) { return "Call that gives access to render tasks."; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallGPURenderTaskData> GPURenderTasksDataCallDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMeshData> CallGPUMeshDataDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMaterialData> CallGPUMaterialDataDescription;
typedef megamol::core::factories::CallAutoDescription<Call3DInteraction> Call3DInteractionDescription;
typedef megamol::core::factories::CallAutoDescription<CallGlTFData> CallGlTFDataDescription;

} // namespace mesh
} // namespace megamol


#endif // !MESH_CALLS_H_INCLUDED
