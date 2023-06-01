/*
 * ClipPlane.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmstd/renderer/ClipPlane.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol::core;


megamol::core::view::ClipPlane::ClipPlane()
        : Module()
        , getClipPlaneSlot("getclipplane", "Provides the clipping plane")
        , plane()
        , cameraSerializer()
        , enableSlot("clip::enable", "Disables or enables the clipping plane")
        , colourSlot("clip::colour", "Defines the colour of the clipping plane")
        , normalSlot("clip::normal", "Defines the normal of the clipping plane")
        , pointSlot("clip::point", "Defines a point in the clipping plane")
        , distSlot("clip::dist", "The plane-origin distance")
        , cameraSlot("clip::camera", "The serialized camera") {

    this->plane.Set(vislib::math::Point<float, 3>(0.0, 0.0f, 0.0f), vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->col[0] = this->col[1] = this->col[2] = 0.5f;
    this->col[3] = 1.0f;

    view::CallClipPlaneDescription ccpd;
    this->getClipPlaneSlot.SetCallback(ccpd.ClassName(), ccpd.FunctionName(0), &ClipPlane::requestPlane);
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->enableSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->enableSlot);

    this->colourSlot << new param::ColorParam(this->col[0], this->col[1], this->col[2], this->col[3]);
    this->MakeSlotAvailable(&this->colourSlot);

    this->normalSlot << new param::Vector3fParam(this->plane.Normal());
    this->MakeSlotAvailable(&this->normalSlot);

    this->pointSlot << new param::Vector3fParam(vislib::math::Vector<float, 3>(this->plane.Point()));
    this->MakeSlotAvailable(&this->pointSlot);

    this->distSlot << new param::FloatParam(-this->plane.Distance(vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f)));
    this->MakeSlotAvailable(&this->distSlot);

    this->cameraSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->cameraSlot);
    this->cameraSlot.Parameter()->SetGUIReadOnly(true);
    this->cameraSlot.Parameter()->SetGUIVisible(false);
}


megamol::core::view::ClipPlane::~ClipPlane() {

    this->Release();
}


bool megamol::core::view::ClipPlane::create() {

    // intentionally empty
    return true;
}


void megamol::core::view::ClipPlane::release() {

    // intentionally empty
}


bool megamol::core::view::ClipPlane::requestPlane(Call& call) {

    megamol::core::view::CallClipPlane* ccp = dynamic_cast<megamol::core::view::CallClipPlane*>(&call);
    if (ccp == nullptr) {
        return false;
    }

    auto camera = ccp->GetCamera();
    this->cameraSerializer.setPrettyMode(false);
    std::string camstring = this->cameraSerializer.serialize(camera);
    this->cameraSlot.Param<param::StringParam>()->SetValue(camstring);

    if (!this->enableSlot.Param<param::BoolParam>()->Value()) {
        // clipping plane is disabled
        return false;
    }

    if (this->colourSlot.IsDirty()) {
        this->colourSlot.ResetDirty();
        this->colourSlot.Param<param::ColorParam>()->Value(this->col[0], this->col[1], this->col[2], this->col[3]);
    }

    if (!this->normalSlot.IsDirty() && !this->pointSlot.IsDirty() && this->distSlot.IsDirty()) {
        this->distSlot.ResetDirty();
        vislib::math::Vector<float, 3> n(this->plane.Normal());
        vislib::math::Vector<float, 3> p(n);
        p.Normalise();
        p *= this->distSlot.Param<param::FloatParam>()->Value();
        this->plane.Set(vislib::math::Point<float, 3>(p.PeekComponents()), n);
        this->pointSlot.Param<param::Vector3fParam>()->SetValue(p, false);

    } else if (this->normalSlot.IsDirty() || this->pointSlot.IsDirty()) {

        this->normalSlot.ResetDirty();
        this->pointSlot.ResetDirty();
        this->distSlot.ResetDirty();

        this->plane.Set(vislib::math::Point<float, 3>(const_cast<float*>(
                            this->pointSlot.Param<param::Vector3fParam>()->Value().PeekComponents())),
            this->normalSlot.Param<param::Vector3fParam>()->Value());

        this->distSlot.Param<param::FloatParam>()->SetValue(
            -this->plane.Distance(vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f)), false);
    }

    auto r = static_cast<unsigned char>(this->col[0] * 255.0f);
    auto g = static_cast<unsigned char>(this->col[1] * 255.0f);
    auto b = static_cast<unsigned char>(this->col[2] * 255.0f);
    auto a = static_cast<unsigned char>(this->col[3] * 255.0f);
    ccp->SetColour(r, g, b, a);
    ccp->SetPlane(this->plane);

    return true;
}
