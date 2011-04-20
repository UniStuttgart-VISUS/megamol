/*
 * TrackerRendererTransform.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TrackerRendererTransform.h"
#include "param/FloatParam.h"
#include "param/Vector3fParam.h"
#include "param/Vector4fParam.h"
#include "view/CallRender3D.h"
#include "vislib/Matrix.h"
#include "vislib/Quaternion.h"
#include "vislib/ShallowQuaternion.h"
#include "vislib/ShallowVector.h"
#include "vislib/Vector.h"
#include <GL/gl.h>

using namespace megamol;
using namespace megamol::trisoup;


/*
 * TrackerRendererTransform::TrackerRendererTransform
 */
TrackerRendererTransform::TrackerRendererTransform(void) : Renderer3DModule(),
        outRenderSlot("outrender", "The slot to call the real renderer"),
        translateSlot("translate", "The translation applied"),
        rotateSlot("rotate", "The rotation applied"),
        scaleSlot("scale", "The scale applied"),
        bboxMinSlot("bbox::min", "The minimum vector of the bounding box"),
        bboxMaxSlot("bbox::max", "The maximum vector of the bounding box") {

    this->outRenderSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->outRenderSlot);

    this->translateSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->MakeSlotAvailable(&this->translateSlot);

    this->rotateSlot << new core::param::Vector4fParam(
        vislib::math::ShallowVector<float, 4>(
            vislib::math::Quaternion<float>().PeekComponents()));
    this->MakeSlotAvailable(&this->rotateSlot);

    this->scaleSlot << new core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->scaleSlot);

    this->bboxMinSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(-3.0f, -3.0f, -3.0f));
    this->MakeSlotAvailable(&this->bboxMinSlot);

    this->bboxMaxSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(3.0f, 3.0f, 3.0f));
    this->MakeSlotAvailable(&this->bboxMaxSlot);

}


/*
 * TrackerRendererTransform::~TrackerRendererTransform
 */
TrackerRendererTransform::~TrackerRendererTransform(void) {
    this->Release();
}


/*
 * TrackerRendererTransform::create
 */
bool TrackerRendererTransform::create(void) {
    // intentionally empty
    return true;
}


/*
 * TrackerRendererTransform::GetCapabilities
 */
bool TrackerRendererTransform::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *inCr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (inCr3d == NULL) return false;
    core::view::CallRender3D *outCr3d = this->outRenderSlot.CallAs<core::view::CallRender3D>();
    if (outCr3d == NULL) return false;

    if ((*outCr3d)(2)) {
        *inCr3d = *outCr3d;
        return true;
    }

    return false;
}


/*
 * TrackerRendererTransform::GetExtents
 */
bool TrackerRendererTransform::GetExtents(core::Call& call) {
    core::view::CallRender3D *inCr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (inCr3d == NULL) return false;

    inCr3d->AccessBoundingBoxes().Clear();
    inCr3d->SetTimeFramesCount(1);

    core::view::CallRender3D *outCr3d = this->outRenderSlot.CallAs<core::view::CallRender3D>();
    if ((outCr3d != NULL) && ((*outCr3d)(1))) {

        // TODO: calculate real clip box ... ok for now

        inCr3d->SetTimeFramesCount(outCr3d->TimeFramesCount());
    }

    const vislib::math::Vector<float, 3>& minV = this->bboxMinSlot.Param<core::param::Vector3fParam>()->Value();
    const vislib::math::Vector<float, 3>& maxV = this->bboxMaxSlot.Param<core::param::Vector3fParam>()->Value();
    inCr3d->AccessBoundingBoxes().SetWorldSpaceBBox(minV.X(), minV.Y(), minV.Z(), maxV.X(), maxV.Y(), maxV.Z());

    return true;
}


/*
 * TrackerRendererTransform::release
 */
void TrackerRendererTransform::release(void) {
    // intentionally empty
}


/*
 * TrackerRendererTransform::Render
 */
bool TrackerRendererTransform::Render(core::Call& call) {
    core::view::CallRender3D *inCr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (inCr3d == NULL) return false;
    core::view::CallRender3D *outCr3d = this->outRenderSlot.CallAs<core::view::CallRender3D>();
    if (outCr3d == NULL) return false;

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();

    const vislib::math::Vector<float, 3>& trans = this->translateSlot.Param<core::param::Vector3fParam>()->Value();
    const vislib::math::Vector<float, 4>& rot = this->rotateSlot.Param<core::param::Vector4fParam>()->Value();
    const float& scale = this->scaleSlot.Param<core::param::FloatParam>()->Value();
    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> rotMat;
    rotMat = vislib::math::ShallowQuaternion<float>(const_cast<float*>(rot.PeekComponents()));
    float rotMatBig[16];
    rotMatBig[0] = rotMat(0, 0);
    rotMatBig[1] = rotMat(1, 0);
    rotMatBig[2] = rotMat(2, 0);
    rotMatBig[3] = 0.0f;
    rotMatBig[4] = rotMat(0, 1);
    rotMatBig[5] = rotMat(1, 1);
    rotMatBig[6] = rotMat(2, 1);
    rotMatBig[7] = 0.0f;
    rotMatBig[8] = rotMat(0, 2);
    rotMatBig[9] = rotMat(1, 2);
    rotMatBig[10] = rotMat(2, 2);
    rotMatBig[11] = 0.0f;
    rotMatBig[12] = 0.0f;
    rotMatBig[13] = 0.0f;
    rotMatBig[14] = 0.0f;
    rotMatBig[15] = 1.0f;

    ::glTranslatef(trans.X(), trans.Y(), trans.Z());
    ::glMultMatrixf(rotMatBig);
    ::glScalef(scale, scale, scale);

    *outCr3d = *inCr3d;
    bool retVal = (*outCr3d)(0);

    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

    return retVal;
}
