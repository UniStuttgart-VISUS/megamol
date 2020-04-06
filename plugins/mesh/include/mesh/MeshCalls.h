/*
 * MeshCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_CALLS_H_INCLUDED
#define MESH_CALLS_H_INCLUDED

#include "mmcore/CallGeneric.h"

#include <memory>
#include "tiny_gltf.h"

#include "3DInteractionCollection.h"
#include "GPUMaterialCollection.h"
#include "GPUMeshCollection.h"
#include "GPURenderTaskCollection.h"
#include "ImageDataAccessCollection.h"
#include "MeshDataAccessCollection.h"

namespace megamol {
namespace mesh {

class MESH_API Call3DInteraction
    : public core::GenericVersionedCall<std::shared_ptr<ThreeDimensionalInteractionCollection>, core::EmptyMetaData> {
public:
    inline Call3DInteraction()
        : GenericVersionedCall<std::shared_ptr<ThreeDimensionalInteractionCollection>, core::EmptyMetaData>() {}
    ~Call3DInteraction() = default;

    static const char* ClassName(void) { return "Call3DInteraction"; }
    static const char* Description(void) { return "Call that transports..."; }
};

class MESH_API CallGlTFData : public core::GenericVersionedCall<std::shared_ptr<tinygltf::Model>, core::EmptyMetaData> {
public:
    inline CallGlTFData() : GenericVersionedCall<std::shared_ptr<tinygltf::Model>, core::EmptyMetaData>() {}
    ~CallGlTFData() = default;

    static const char* ClassName(void) { return "CallGlTFData"; }
    static const char* Description(void) { return "Call that gives access to a loaded gltf model."; }
};

class MESH_API CallGPUMaterialData
    : public core::GenericVersionedCall<std::shared_ptr<GPUMaterialCollecton>, core::EmptyMetaData> {
public:
    CallGPUMaterialData() : GenericVersionedCall<std::shared_ptr<GPUMaterialCollecton>, core::EmptyMetaData>() {}
    ~CallGPUMaterialData() = default;

    static const char* ClassName(void) { return "CallGPUMaterialData"; }
    static const char* Description(void) { return "Call that gives access to material data stored on the GPU."; }
};

class MESH_API CallGPUMeshData
    : public core::GenericVersionedCall<std::shared_ptr<GPUMeshCollection>, core::Spatial3DMetaData> {
public:
    CallGPUMeshData() : GenericVersionedCall<std::shared_ptr<GPUMeshCollection>, core::Spatial3DMetaData>() {}
    ~CallGPUMeshData() = default;

    static const char* ClassName(void) { return "CallGPUMeshData"; }
    static const char* Description(void) {
        return "Call that gives access to meshes stored in batches on the GPU for rendering.";
    }
};

class MESH_API CallGPURenderTaskData
    : public core::GenericVersionedCall<std::shared_ptr<GPURenderTaskCollection>, core::Spatial3DMetaData> {
public:
    CallGPURenderTaskData()
        : GenericVersionedCall<std::shared_ptr<GPURenderTaskCollection>, core::Spatial3DMetaData>() {}
    ~CallGPURenderTaskData(){};

    static const char* ClassName(void) { return "CallGPURenderTaskData"; }
    static const char* Description(void) { return "Call that gives access to render tasks."; }
};

class MESH_API CallMesh : public core::GenericVersionedCall<std::shared_ptr<MeshDataAccessCollection>, core::Spatial3DMetaData> {
public:
    CallMesh() : GenericVersionedCall<std::shared_ptr<MeshDataAccessCollection>, core::Spatial3DMetaData>() {}
    ~CallMesh(){};

    static const char* ClassName(void) { return "CallMesh"; }
    static const char* Description(void) { return "Call that gives access to CPU-side mesh data."; }
};

class MESH_API CallImage
    : public core::GenericVersionedCall<std::shared_ptr<ImageDataAccessCollection>, core::EmptyMetaData> {
public:
    CallImage() : GenericVersionedCall<std::shared_ptr<ImageDataAccessCollection>, core::EmptyMetaData>() {}
    ~CallImage(){};

    static const char* ClassName(void) { return "CallImage"; }
    static const char* Description(void) { return "Call that gives access to CPU-side image data."; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallGPURenderTaskData> GPURenderTasksDataCallDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMeshData> CallGPUMeshDataDescription;
typedef megamol::core::factories::CallAutoDescription<CallGPUMaterialData> CallGPUMaterialDataDescription;
typedef megamol::core::factories::CallAutoDescription<Call3DInteraction> Call3DInteractionDescription;
typedef megamol::core::factories::CallAutoDescription<CallGlTFData> CallGlTFDataDescription;
typedef megamol::core::factories::CallAutoDescription<CallMesh> CallMeshDescription;
typedef megamol::core::factories::CallAutoDescription<CallImage> CallImageDescription;

} // namespace mesh
} // namespace megamol


#endif // !MESH_CALLS_H_INCLUDED
