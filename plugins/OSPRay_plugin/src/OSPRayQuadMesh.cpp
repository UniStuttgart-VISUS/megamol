/*
 * OSPRayQuadMesh.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayQuadMesh.h"
#include "geometry_calls/CallTriMeshData.h"
#include "vislib/sys/Log.h"
#include "mmcore/BoundingBoxes_2.h"


using namespace megamol::ospray;


OSPRayQuadMesh::OSPRayQuadMesh(void)
    : AbstractOSPRayStructure()
    , getMeshDataSlot("getMeshData", "Connects to the data source") {

    this->getMeshDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->getMeshDataSlot);
}


bool OSPRayQuadMesh::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getMeshDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {
        auto meta_data = cm->getMetaData();
        this->structureContainer.dataChanged = false;
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);
        if (!(*cm)(1)) return false;
        if (!(*cm)(0)) return false;
        meta_data = cm->getMetaData();
        if (cm->hasUpdate() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
            this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
            this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
            this->extendContainer.isValid = true;
            this->structureContainer.mesh = cm->getData();
        }
    }

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::QUADS;


    return true;
}


OSPRayQuadMesh::~OSPRayQuadMesh() { this->Release(); }

bool OSPRayQuadMesh::create() { return true; }

void OSPRayQuadMesh::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayQuadMesh::InterfaceIsDirty() {
        return false;
}


bool OSPRayQuadMesh::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getMeshDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {

        if (!(*cm)(1)) return false;
        auto meta_data = cm->getMetaData();
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);

        this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
        this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
        this->extendContainer.isValid = true;

    } 
    return true;
}