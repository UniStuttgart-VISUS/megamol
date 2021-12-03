/*
 * MeshCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_CALLS_H_INCLUDED
#define MESH_CALLS_H_INCLUDED

#include "mmcore/CallGeneric.h"

#include "tiny_gltf.h"
#include <memory>
#include <vector>

#include "3DInteractionCollection.h"
#include "ImageDataAccessCollection.h"
#include "MeshDataAccessCollection.h"

namespace megamol {
namespace mesh {

class Call3DInteraction : public core::GenericVersionedCall<std::shared_ptr<ThreeDimensionalInteractionCollection>,
                              core::EmptyMetaData> {
public:
    inline Call3DInteraction()
            : GenericVersionedCall<std::shared_ptr<ThreeDimensionalInteractionCollection>, core::EmptyMetaData>() {}
    ~Call3DInteraction() = default;

    static const char* ClassName(void) {
        return "Call3DInteraction";
    }
    static const char* Description(void) {
        return "Call that transports...";
    }
};

class CallGlTFData : public core::GenericVersionedCall<std::pair<std::string, std::shared_ptr<tinygltf::Model>>,
                         core::EmptyMetaData> {
public:
    inline CallGlTFData()
            : GenericVersionedCall<std::pair<std::string, std::shared_ptr<tinygltf::Model>>, core::EmptyMetaData>() {}
    ~CallGlTFData() = default;

    static const char* ClassName(void) {
        return "CallGlTFData";
    }
    static const char* Description(void) {
        return "Call that gives access to a loaded gltf model.";
    }
};

class CallMesh : public core::GenericVersionedCall<std::shared_ptr<MeshDataAccessCollection>, core::Spatial3DMetaData> {
public:
    CallMesh() : GenericVersionedCall<std::shared_ptr<MeshDataAccessCollection>, core::Spatial3DMetaData>() {}
    ~CallMesh(){};

    static const char* ClassName(void) {
        return "CallMesh";
    }
    static const char* Description(void) {
        return "Call that gives access to CPU-side mesh data.";
    }
};

class CallImage : public core::GenericVersionedCall<std::shared_ptr<ImageDataAccessCollection>, core::EmptyMetaData> {
public:
    CallImage() : GenericVersionedCall<std::shared_ptr<ImageDataAccessCollection>, core::EmptyMetaData>() {}
    ~CallImage(){};

    static const char* ClassName(void) {
        return "CallImage";
    }
    static const char* Description(void) {
        return "Call that gives access to CPU-side image data.";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<Call3DInteraction> Call3DInteractionDescription;
typedef megamol::core::factories::CallAutoDescription<CallGlTFData> CallGlTFDataDescription;
typedef megamol::core::factories::CallAutoDescription<CallMesh> CallMeshDescription;
typedef megamol::core::factories::CallAutoDescription<CallImage> CallImageDescription;

} // namespace mesh
} // namespace megamol


#endif // !MESH_CALLS_H_INCLUDED
