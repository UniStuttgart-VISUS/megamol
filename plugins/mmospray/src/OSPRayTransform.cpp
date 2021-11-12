/*
* OSPRayTransform.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayTransform.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "glm/matrix.hpp"


namespace megamol {
namespace ospray {


OSPRayTransform::OSPRayTransform(void) : 
    _pos("pos", ""),
    _rot("rot", ""),
    _scale("scale", ""),
    _deployTransformationSlot("deployTransformationSlot", "")
 {

    this->_scale << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->_pos << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->_rot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));

    this->MakeSlotAvailable(&this->_scale);
    this->MakeSlotAvailable(&this->_rot);
    this->MakeSlotAvailable(&this->_pos);

    this->_transformationContainer.isValid = true;

    this->_deployTransformationSlot.SetCallback(CallOSPRayTransformation::ClassName(),
        CallOSPRayTransformation ::FunctionName(0), &OSPRayTransform::getTransformationCallback);
    this->MakeSlotAvailable(&this->_deployTransformationSlot);

}

OSPRayTransform::~OSPRayTransform(void) {
    this->Release();
}

bool OSPRayTransform::getTransformationCallback(core::Call &call) {
    CallOSPRayTransformation* tc_in = dynamic_cast<CallOSPRayTransformation*>(&call);

    if (tc_in != NULL) {
        this->readParams();
        tc_in->setTransformationContainer(std::make_shared<OSPRayTransformationContainer>(this->_transformationContainer));
    }

    if (this->InterfaceIsDirty()) {
        tc_in->setDirty();
    }

    return true;
}

void OSPRayTransform::readParams() {

    auto pos = this->_pos.Param<core::param::Vector3fParam>();
    _transformationContainer.pos = pos->getArray();

    auto rot = this->_rot.Param<core::param::Vector3fParam>()->getArray();
    auto scale = this->_scale.Param<core::param::Vector3fParam>()->getArray();
    
    glm::mat3x3 rot_x = {{1, 0, 0}, {0, std::cos(rot[0]), -std::sin(rot[0])}, {0, std::sin(rot[0]), std::cos(rot[0])}};
    glm::mat3x3 rot_y = {{std::cos(rot[1]), 0, std::sin(rot[1])},{0,1,0},{-std::sin(rot[1]), 0, std::cos(rot[1])}};
    glm::mat3x3 rot_z = {{std::cos(rot[2]), -std::sin(rot[2]), 0},{std::sin(rot[2]), std::cos(rot[2]), 0},{0,0,1}};

    glm::mat3x3 scaleMX = {{scale[0],0,0},{0,scale[1],0},{0,0,scale[2]}};

    auto affine_trafo = (scaleMX * rot_x * rot_y * rot_z);

    _transformationContainer.MX[0][0] = affine_trafo[0][0];
    _transformationContainer.MX[1][0] = affine_trafo[1][0];
    _transformationContainer.MX[2][0] = affine_trafo[2][0];
    _transformationContainer.MX[0][1] = affine_trafo[0][1];
    _transformationContainer.MX[1][1] = affine_trafo[1][1];
    _transformationContainer.MX[2][1] = affine_trafo[2][1];
    _transformationContainer.MX[0][2] = affine_trafo[0][2];
    _transformationContainer.MX[1][2] = affine_trafo[1][2];
    _transformationContainer.MX[2][2] = affine_trafo[2][2];

}

bool OSPRayTransform::InterfaceIsDirty() {
    if (this->_pos.IsDirty() ||
        this->_rot.IsDirty() ||
        this->_scale.IsDirty()) {
        this->_pos.ResetDirty();
        this->_rot.ResetDirty();
        this->_scale.ResetDirty();
        return true;
    } else {
        return false;
    }
}

} // namespace ospray
} // namespace megamol