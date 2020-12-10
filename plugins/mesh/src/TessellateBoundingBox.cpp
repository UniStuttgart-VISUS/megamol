/*
 * TessellateBoundingBox.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "TessellateBoundingBox.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D_2.h"

megamol::mesh::TessellateBoundingBox::TessellateBoundingBox(void)
        : _bounding_box_rhs_slot("", "")
        , _mesh_lhs_slot("", "")
        , _subdiv_slot("", "")
        , _face_type_slot("", "") {

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "Trianges");
    ep->SetTypePair(1, "Quads");
    this->_face_type_slot << ep;
    this->MakeSlotAvailable(&this->_face_type_slot);

    this->_mesh_lhs_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &TessellateBoundingBox::getData);
    this->_mesh_lhs_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &TessellateBoundingBox::getMetaData);
    this->MakeSlotAvailable(&this->_mesh_lhs_slot);

    this->_bounding_box_rhs_slot.SetCompatibleCall<core::view::CallRender3D_2Description>();
    this->MakeSlotAvailable(&this->_bounding_box_rhs_slot);
}

megamol::mesh::TessellateBoundingBox::~TessellateBoundingBox(void) {
    this->Release();
}

bool megamol::mesh::TessellateBoundingBox::create() {
    return true;
}

void megamol::mesh::TessellateBoundingBox::release() {}

bool megamol::mesh::TessellateBoundingBox::InterfaceIsDirty() {
    return false;
}

bool megamol::mesh::TessellateBoundingBox::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) {
        return false;
    }

    auto meta_data = cm->getMetaData();
        
    core::view::CallRender3D_2* bboxc = this->_bounding_box_rhs_slot.CallAs<core::view::CallRender3D_2>();
    if (bboxc != nullptr) {
        // get extends from render call
        bboxc->SetTime(meta_data.m_frame_ID); // time equal frame ID ? help!
        if (!(*bboxc)(core::view::AbstractCallRender::FnGetExtents)) {
            return false;
        }

        // put metadata in mesh call
        meta_data.m_bboxs = bboxc->AccessBoundingBoxes();
        meta_data.m_frame_cnt = bboxc->TimeFramesCount();
    }

    cm->setMetaData(meta_data);

    return true;
}

bool megamol::mesh::TessellateBoundingBox::getData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) {
        return false;
    }

    auto meta_data = cm->getMetaData();

    core::view::CallRender3D_2* bboxc = dynamic_cast<core::view::CallRender3D_2*>(&call);
    if (bboxc != nullptr) {
        // get extends from render call
        bboxc->SetTime(meta_data.m_frame_ID); // time equal frame ID ? help!
        if (!(*bboxc)(core::view::AbstractCallRender::FnGetExtents)) {
            return false;
        }

        // put metadata in mesh call
        meta_data.m_bboxs = bboxc->AccessBoundingBoxes();
        meta_data.m_frame_cnt = bboxc->TimeFramesCount();



        //TODO get bounding box corners

        //TODO tessellate each face
    }


    return true;
}
