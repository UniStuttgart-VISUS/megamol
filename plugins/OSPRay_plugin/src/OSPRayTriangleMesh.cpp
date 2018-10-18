/*
 * OSPRayTriangleMesh.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include <functional>
#include "OSPRayTriangleMesh.h"
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "vislib/sys/Log.h"


using namespace megamol::ospray;


OSPRayTriangleMesh::OSPRayTriangleMesh(void)
    : AbstractOSPRayStructure()
    , getDataSlot("getdata", "Connects to the data source")
    ,

    objectID("objectID", "Switches between objects") {

    this->getDataSlot.SetCompatibleCall<megamol::geocalls::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);


    this->objectID << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->objectID);
}


bool OSPRayTriangleMesh::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::geocalls::CallTriMeshData* cd = this->getDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    unsigned int objectCount = cd->Count();


    if (this->objectID.Param<core::param::IntParam>()->Value() > (objectCount - 1)) {
        this->objectID.Param<core::param::IntParam>()->SetValue(0);
    }


    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;

    std::vector<float> vertexD;
    std::vector<float> colorD;
    std::vector<float> normalD;
    std::vector<float> texD;
    std::vector<unsigned int> indexD;


    const geocalls::CallTriMeshData::Mesh& obj = cd->Objects()[this->objectID.Param<core::param::IntParam>()->Value()];
    unsigned int triangleCount = obj.GetTriCount();
    unsigned int vertexCount = obj.GetVertexCount();

    // check vertex data type
    switch (obj.GetVertexDataType()) {
    case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
        auto vertexPointer = obj.GetVertexPointerFloat();
        vertexD.resize(obj.GetVertexCount() * 3);
        vertexD.assign(vertexPointer, vertexPointer + obj.GetVertexCount() * 3);
        //std::transform(vertexD.begin(), vertexD.end(), vertexD.begin(),
        //    std::bind(std::multiplies<float>(), std::placeholders::_1, 3));
        break;
        // case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
        // ospray does not support double vectors/arrays
        // OSPData vertexdata = ospNewData(obj.GetVertexCount, OSP_DOUBLE3, obj.GetVertexPointerFloat());
        //    break;
    }

    // check normal pointer
    if (obj.HasNormalPointer() != NULL) {
        switch (obj.GetNormalDataType()) {
        case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
            auto normalPointer = obj.GetNormalPointerFloat();
            normalD.resize(obj.GetVertexCount() * 3);
            normalD.assign(normalPointer, normalPointer + obj.GetVertexCount() * 3);
            break;
            // case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
            //    break;
        }
    }

    // check colorpointer and convert to rgba
    if (obj.HasColourPointer() != NULL) {
        switch (obj.GetColourDataType()) {
        case geocalls::CallTriMeshData::Mesh::DT_BYTE:
            colorD.reserve(vertexCount * 4);
            for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                colorD.push_back((float)obj.GetColourPointerByte()[i] / 255.0f);
                if ((i + 1) % 3 == 0) {
                    colorD.push_back(1.0f);
                }
            }
            break;
        case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
            // TODO: not tested
            colorD.reserve(vertexCount * 4);
            for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                colorD.push_back(obj.GetColourPointerFloat()[i]);
                if ((i + 1) % 3 == 0) {
                    colorD.push_back(1.0f);
                }
            }
            break;
            // case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
            //    break;
        }
    }


    // check texture array
    if (obj.HasTextureCoordinatePointer() != NULL) {
        switch (obj.GetTextureCoordinateDataType()) {
        case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
            auto texPointer = obj.GetTextureCoordinatePointerFloat();
            texD.resize(obj.GetTriCount() * 2);
            texD.assign(texPointer, texPointer + obj.GetTriCount() * 2);
            break;
            // case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
            //    break;
        }
    }

    // check index pointer
    if (obj.HasTriIndexPointer() != NULL) {
        switch (obj.GetTriDataType()) {
            // case trisoup::CallTriMeshData::Mesh::DT_BYTE:
            //    break;
            // case trisoup::CallTriMeshData::Mesh::DT_UINT16:
            //    break;
        case geocalls::CallTriMeshData::Mesh::DT_UINT32:
            auto indexPointer = obj.GetTriIndexPointerUInt32();
            indexD.resize(obj.GetTriCount() * 3);
            indexD.assign(indexPointer, indexPointer + obj.GetTriCount() * 3);
            break;
        }
    }

    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::TRIANGLES;
    this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(vertexD));
    this->structureContainer.colorData = std::make_shared<std::vector<float>>(std::move(colorD));
    this->structureContainer.normalData = std::make_shared<std::vector<float>>(std::move(normalD));
    this->structureContainer.texData = std::make_shared<std::vector<float>>(std::move(texD));
    this->structureContainer.indexData = std::make_shared<std::vector<uint32_t>>(std::move(indexD));
    this->structureContainer.vertexCount = vertexCount;
    this->structureContainer.triangleCount = triangleCount;

    return true;
}


OSPRayTriangleMesh::~OSPRayTriangleMesh() { this->Release(); }

bool OSPRayTriangleMesh::create() { return true; }

void OSPRayTriangleMesh::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayTriangleMesh::InterfaceIsDirty() {
    if (this->objectID.IsDirty()) {
        this->objectID.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayTriangleMesh::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::geocalls::CallTriMeshData* cd = this->getDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();

    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }

    if (!(*cd)(1)) return false;

    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}