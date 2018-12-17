/*
 * MuxRenderer3D.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/RendererRegistration.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"

namespace megamol {
namespace core {
namespace view {

/*
* MuxRenderer3D<T>::MuxRenderer3D
*/
RendererRegistration::RendererRegistration(void) : Renderer3DModule(), frameCnt(0), bboxs(), scale(1.0f),
    rendererSlot("renderer", "slot for the original renderer"),
    scaleXSlot("scaleX", ""),
    scaleYSlot("scaleY", ""),
    scaleZSlot("scaleZ", ""),
    translateXSlot("translateX", ""),
    translateYSlot("translateY", ""),
    translateZSlot("translateZ", ""),
    rotateXSlot("rotateX", ""),
    rotateYSlot("rotateY", ""),
    rotateZSlot("rotateZ", "") {

    this->rendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);

    this->scaleXSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scaleXSlot);
    this->scaleYSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scaleYSlot);
    this->scaleZSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scaleZSlot);

    this->translateXSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->translateXSlot);
    this->translateYSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->translateYSlot);
    this->translateZSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->translateZSlot);

    this->rotateXSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->rotateXSlot);
    this->rotateYSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->rotateYSlot);
    this->rotateZSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->rotateZSlot);

    //this->rendererSlot[i] = new CallerSlot(name, desc);
    //this->rendererSlot[i]->template SetCompatibleCall<CallRender3DDescription>();
    //this->MakeSlotAvailable(this->rendererSlot[i]);

    //vislib::StringA name, desc;
    //for (unsigned int i = 0; i < T; i++) {

    //    name.Format("renderer%u", i + 1);
    //    desc.Format("Outgoing renderer #%u", i + 1);

    //    name += "active";
    //    desc.Format("De-/Activates outgoing renderer #%u", i + 1);
    //    this->rendererActiveSlot[i] = new param::ParamSlot(name, desc);
    //    this->rendererActiveSlot[i]->SetParameter(new param::BoolParam(true));
    //    this->MakeSlotAvailable(this->rendererActiveSlot[i]);
    //}
}


/*
* MuxRenderer3D<T>::~MuxRenderer3D
*/
RendererRegistration::~RendererRegistration(void) {
    this->Release();
}


/*
* MuxRenderer3D<T>::create
*/
bool RendererRegistration::create(void) {
    // intentionally empty
    return true;
}


/*
* MuxRenderer3D<T>::release
*/
void RendererRegistration::release(void) {
    // intentionally empty
}


/*
* MuxRenderer3D<T>::GetExtents
*/
bool RendererRegistration::GetExtents(Call& call) {
    CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
    if (cr3d == NULL) return false;

    this->bboxs.Clear();
    this->frameCnt = 0;
    CallRender3D *oc = this->rendererSlot.CallAs<CallRender3D>();
    if ((oc == NULL) || (!(*oc)(core::view::AbstractCallRender::FnGetExtents))) return false;
    if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
        this->bboxs.SetObjectSpaceBBox(oc->AccessBoundingBoxes().ObjectSpaceBBox());
    } else if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        this->bboxs.SetObjectSpaceBBox(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
    } else {
        this->bboxs.SetObjectSpaceBBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }
    if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        this->bboxs.SetObjectSpaceClipBox(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
    } else if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
        this->bboxs.SetObjectSpaceClipBox(oc->AccessBoundingBoxes().ObjectSpaceBBox());
    } else {
        this->bboxs.SetObjectSpaceClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }
    this->frameCnt = vislib::math::Max(this->frameCnt, oc->TimeFramesCount());
    if (this->frameCnt == 0) {
        this->frameCnt = 1;
        //this->scale = 1.0f;
        this->bboxs.Clear();
    } else {
        this->scale = 1.0f / this->bboxs.ObjectSpaceBBox().LongestEdge();
        this->bboxs.MakeScaledWorld(scale);
    }

    cr3d->SetTimeFramesCount(this->frameCnt);
    cr3d->AccessBoundingBoxes() = this->bboxs;

    return true;
}


/*
* MuxRenderer3D<T>::Render
*/
bool RendererRegistration::Render(Call& call) {
    CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
    vislib::SmartPtr<vislib::graphics::CameraParameters> camParams
        = new vislib::graphics::CameraParamsStore();
    if (cr3d == NULL) return false;

    if (this->frameCnt == 0) {
        this->GetExtents(call);
    }

    CallRender3D *oc = this->rendererSlot.CallAs<CallRender3D>();
    if (oc == NULL) return false;
    *oc = *cr3d;
    if (!(*oc)(core::view::AbstractCallRender::FnGetExtents)) return false;

    // Back translation ocWS -> ocOS
    //float sx, sy, sz, tx, ty, tz;
    //const vislib::math::Cuboid<float>& ocWS = oc->AccessBoundingBoxes().WorldSpaceBBox();
    //const vislib::math::Cuboid<float>& ocOS = oc->AccessBoundingBoxes().ObjectSpaceBBox();

    //sx = ocOS.Width() / ocWS.Width();
    //sy = ocOS.Height() / ocWS.Height();
    //sz = ocOS.Depth() / ocWS.Depth();

    //tx = ocWS.Left() * sx - ocOS.Left();
    //ty = ocWS.Bottom() * sy - ocOS.Bottom();
    //tz = ocWS.Back() * sz - ocOS.Back();

    // We clamp the time to the range of the individual renderers
    int octfc = oc->TimeFramesCount();
    *oc = *cr3d;
    oc->SetTime(vislib::math::Min<float>(cr3d->Time(), static_cast<float>(octfc - 1)));

    camParams->CopyFrom(cr3d->GetCameraParameters());
    oc->SetCameraParameters(camParams);
    vislib::math::Point<float, 3> p = camParams->Position();
    vislib::math::Point<float, 3> l = camParams->LookAt();
    //p.Set((p.X() / this->scale - tx) / sx,
    //    (p.Y() / this->scale - ty) / sy,
    //    (p.Z() / this->scale - tz) / sz);
    //l.Set((l.X() / this->scale - tx) / sx,
    //    (l.Y() / this->scale - ty) / sy,
    //    (l.Z() / this->scale - tz) / sz);

    camParams->SetView(p, l, camParams->Up());

    ::glMatrixMode(GL_MODELVIEW);

    float aMatrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, aMatrix);

    ::glPushMatrix();

    glGetFloatv(GL_MODELVIEW_MATRIX, aMatrix);

    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> scaleMat;
    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> transMat;
    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> rotMat;
    scaleMat.SetIdentity();
    scaleMat.SetAt(0, 0, this->scaleXSlot.Param<param::FloatParam>()->Value());
    scaleMat.SetAt(1, 1, this->scaleYSlot.Param<param::FloatParam>()->Value());
    scaleMat.SetAt(2, 2, this->scaleZSlot.Param<param::FloatParam>()->Value());
    transMat.SetAt(0, 3, this->translateXSlot.Param<param::FloatParam>()->Value());
    transMat.SetAt(1, 3, this->translateYSlot.Param<param::FloatParam>()->Value());
    transMat.SetAt(2, 3, this->translateZSlot.Param<param::FloatParam>()->Value());
    
    vislib::math::Quaternion<float> qx(this->rotateXSlot.Param<param::FloatParam>()->Value(), vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    vislib::math::Quaternion<float> qy(this->rotateYSlot.Param<param::FloatParam>()->Value(), vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    vislib::math::Quaternion<float> qz(this->rotateZSlot.Param<param::FloatParam>()->Value(), vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));

    rotMat = qx * qy * qz;

    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> theMat;
    theMat = scaleMat * transMat * rotMat;

    //glLoadMatrixf(theMat.PeekComponents());
    //glLoadMatrixf(aMatrix);
    glMultMatrixf(theMat.PeekComponents());

    (*oc)(AbstractCallRender::FnRender);

    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

    return true;
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */