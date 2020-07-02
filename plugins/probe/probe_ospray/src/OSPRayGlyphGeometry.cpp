/*
 * OSPRayGlyphGeometry.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayGlyphGeometry.h"
#include "mesh/MeshCalls.h"


namespace megamol {
namespace probe {

OSPRayGlyphGeometry::~OSPRayGlyphGeometry(void) {
    this->Release();
}

OSPRayGlyphGeometry::OSPRayGlyphGeometry(void) : 
    AbstractOSPRayStructure()
    , _get_mesh_slot("getMesh","")
    , _get_texture_slot("getTexture","") 
{
    this->_get_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_get_mesh_slot);

    this->_get_texture_slot.SetCompatibleCall<mesh::CallImageDescription>();
    this->MakeSlotAvailable(&this->_get_texture_slot);

    //this->SetSlotUnavailable(&this->getMaterialSlot);
}

bool OSPRayGlyphGeometry::create() { return true; }

void OSPRayGlyphGeometry::release() {}


bool OSPRayGlyphGeometry::readData(core::Call& call) {
    
    // read Data, calculate  shape parameters, fill data vectors
    ospray::CallOSPRayStructure* os = dynamic_cast<ospray::CallOSPRayStructure*>(&call);
    auto cm = this->_get_mesh_slot.CallAs<mesh::CallMesh>();

    auto ctex = this->_get_texture_slot.CallAs<mesh::CallImage>();

    if (cm == nullptr) return false;
    if (ctex == nullptr) return false;

    auto meta_data = cm->getMetaData();
    auto tex_meta_data = ctex->getMetaData();
    this->structureContainer.dataChanged = false;
    if (os->getTime() > meta_data.m_frame_cnt) {
        meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
    } else {
        meta_data.m_frame_ID = os->getTime();
    }

    cm->setMetaData(meta_data);

    if (!(*cm)(1)) return false;
    if (!(*cm)(0)) return false;

    if (!(*ctex)(1)) return false;
    if (!(*ctex)(0)) return false;

    meta_data = cm->getMetaData();

    if (cm->hasUpdate() || ctex->hasUpdate() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;

        // revalidate the boundingbox
        this->extendContainer.boundingBox->SetBoundingBox(meta_data.m_bboxs.BoundingBox());
        this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
        this->extendContainer.isValid = true;

        // Write stuff into the structureContainer
        this->structureContainer.type = ospray::structureTypeEnum::GEOMETRY;
        this->structureContainer.geometryType = ospray::geometryTypeEnum::TRIANGLES;
        this->structureContainer.mesh = cm->getData();
        this->structureContainer.mesh_textures = ctex->getData();
        this->structureContainer.materialChanged = false;
    }

     return true;
}

bool OSPRayGlyphGeometry::getExtends(core::Call& call) {
    ospray::CallOSPRayStructure* os = dynamic_cast<ospray::CallOSPRayStructure*>(&call);
    auto cm = this->_get_mesh_slot.CallAs<mesh::CallMesh>();

    if (cm == NULL) return false;
    auto meta_data = cm->getMetaData();
    meta_data.m_frame_ID = os->getTime();
    cm->setMetaData(meta_data);
    (*cm)(1);
    meta_data = cm->getMetaData();


    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(meta_data.m_bboxs.BoundingBox());
    this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
    this->extendContainer.isValid = true;

    return true;
}

bool OSPRayGlyphGeometry::InterfaceIsDirty() { return false; }

} // namespace probe
} // namespace megamol

