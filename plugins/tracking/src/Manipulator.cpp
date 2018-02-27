/*
 * Manipulator.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Manipulator.h"

#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"

#include "vislib/Trace.h"


/*
 * megamol::vrpnModule::Manipulator::Manipulator
 */
megamol::vrpnModule::Manipulator::Manipulator(void) :
        isRotating(false), isTranslating(false), isZooming(false),
        paramInvertRotate("manipulator::invertRotate", "Inverts the rotation."),
        paramInvertZoom("manipulator::invertZoom", "Inverts the zoom direction."),
        paramRotateButton("manipulator::rotateButton", "The button that must be pressed for rotation."),
        paramSingleInteraction("manipulator::singleInteraction", "Disables multiple interactions at the same time."),
        paramTranslateButton("manipulator::translateButton", "The button that must be pressed for translation."),
        paramTranslateSpeed("manipulator::translateSpeed", "The translation speed."),
        paramZoomButton("manipulator::zoomButton", "The button that must be pressed for dolly zoom."),
        paramZoomSpeed("manipulator::zoomSpeed", "Transformation of distance to zoom speed.") {
    this->paramInvertRotate << new core::param::BoolParam(true);
    this->params.push_back(&this->paramInvertRotate);

    this->paramInvertZoom << new core::param::BoolParam(true);
    this->params.push_back(&this->paramInvertZoom);

    this->paramRotateButton << new core::param::IntParam(0, 0, 5);
    this->params.push_back(&this->paramRotateButton);

    this->paramSingleInteraction << new core::param::BoolParam(false);
    this->params.push_back(&this->paramSingleInteraction);

    this->paramTranslateButton << new core::param::IntParam(1, 0, 5);
    //this->params.push_back(&this->paramTranslateButton);

    this->paramTranslateSpeed << new core::param::FloatParam(1.0f);
    //this->params.push_back(&this->paramTranslateSpeed);

    this->paramZoomButton << new core::param::IntParam(1, 0, 5);
    this->params.push_back(&this->paramZoomButton);

    this->paramZoomSpeed << new core::param::FloatParam(1.0f);
    this->params.push_back(&this->paramZoomSpeed);

    // Ensure that the initial position is invalid.
    this->curOrientation.Set(0.0f, 0.0f, 0.0f, 0.0f);
    this->curPosition.Set(0.0f, 0.0f, 0.0f);
}


/*
 * megamol::vrpnModule::Manipulator::~Manipulator
 */
megamol::vrpnModule::Manipulator::~Manipulator(void) {
}


/*
 * megamol::vrpnModule::Manipulator::OnButtonChanged
 */
void megamol::vrpnModule::Manipulator::OnButtonChanged(const int button, const bool isPressed) {
    using megamol::core::param::IntParam;

    /* Detect reconfiguration of button mapping. */

    if (this->paramRotateButton.IsDirty()) {
        this->isRotating = false;
        this->isTranslating = false;
        this->isZooming = false;
        this->paramRotateButton.ResetDirty();
    }

    if (this->paramTranslateButton.IsDirty()) {
        this->isRotating = false;
        this->isTranslating = false;
        this->isZooming = false;
        this->paramTranslateButton.ResetDirty();
    }

    if (this->paramZoomButton.IsDirty()) {
        this->isRotating = false;
        this->isTranslating = false;
        this->isZooming = false;
        this->paramZoomButton.ResetDirty();
    }

    /* Remember initial configuration and determine what to do. */
    if (isPressed) {
        // Preserve the current camera system as we need it later to perform
        // incremental transformations.
        this->startCamLookAt = this->camParams->LookAt();
        this->startCamPosition = this->camParams->Position();
        this->startCamUp = this->camParams->Up();

        // Relative orientation of the stick wrt. the current camera coordinate
        // system. This is required to align the interaction device with the
        // current view later on.
        auto q1 = Manipulator::xform(VectorType(0, 0, 1), this->startCamPosition - this->startCamLookAt);
        auto q2 = Manipulator::xform(q1 * VectorType(0, 1, 0), this->startCamUp);
        this->startRelativeOrientation = q2 * q1;

        this->startOrientation = this->curOrientation;
        this->startPosition = this->curPosition;
    }

    int rotateButton = this->paramRotateButton.Param<IntParam>()->Value();
    int translateButton = this->paramTranslateButton.Param<IntParam>()->Value();
    int zoomButton = this->paramZoomButton.Param<IntParam>()->Value();
    translateButton = -1;    // Disable translate

    if (button == rotateButton) {
        //this->startOrientation = this->curOrientation;
        this->isRotating = isPressed;
    } 

    if (button == translateButton) {
        //this->startPosition = this->curPosition;
        this->isTranslating = isPressed;
    }

    if (button == zoomButton) {
        //this->startPosition = this->curPosition;
        this->isZooming = isPressed;
    }
}


/*
 * megamol::vrpnModule::Manipulator::ApplyTransformations
 */
void megamol::vrpnModule::Manipulator::ApplyTransformations(void) {
    using namespace core::param;

    int xact = 0;

    if (this->isRotating) {
        ++xact;
    }
    if (this->isTranslating) {
        ++xact;
    }
    if (this->isZooming) {
        ++xact;
    }

    if (this->paramSingleInteraction.Param<BoolParam>()->Value() && (xact > 1)) {
        xact = 0;
    }

    if ((xact > 0) && (this->camParams != nullptr)) {
        //PointType camPos = this->camParams->Position();
        //PointType lookAt = this->camParams->LookAt();
        //VectorType antiLook = camPos - lookAt;
        //VectorType right = this->camParams->Right();
        //VectorType up = this->camParams->Up();

        if (this->isRotating) {
            VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Apply 6DOF rotation.\n");

            // Compute relative rotation since button was pressed.
            auto quat = this->startOrientation.Inverse();
            quat = this->curOrientation * quat;
            quat.Normalise();

            // Align interaction with the original camera system.
            auto relConj = this->startRelativeOrientation;
            relConj.Conjugate();
            quat = this->startRelativeOrientation * quat * relConj;

            // Optionally, apply the Reina factor.
            if (this->paramInvertRotate.Param<BoolParam>()->Value()) {
                quat.Invert();
            } else {
                quat.Normalise();
            }

            // Apply rotation.
            VectorType startView = this->startCamPosition - this->startCamLookAt;
            VectorType up = quat * this->startCamUp;
            VectorType view = quat * startView;

            // Apply new view parameters.
            this->camParams->SetView(this->startCamLookAt + view, this->startCamLookAt, up);
        }

        if (this->isTranslating) {
            VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Apply 6DOF translation.\n");
#if 0
            // Compute relative movement of tracker in physical space.
            auto delta = this->curPosition - this->startPosition;

            // Scale into virtual space.
            delta *= this->paramTranslateSpeed.Param<FloatParam>()->Value();

            // Align interaction with the original camera system.
            delta = this->startRelativeOrientation * delta;

            // Apply new position.
            this->camParams->SetPosition(this->startCamLookAt + delta);
            this->camParams->SetLookAt(this->startCamLookAt + delta);
#endif
        }

        if (this->isZooming) {
            VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Apply 6DOF zoom.\n");

            // Compute relative movement of tracker in physical space.
            auto diff = this->curPosition - this->startPosition;

            // Compute the distance in virtual space that we move the camera.
            auto delta = diff.Z() * this->paramZoomSpeed.Param<FloatParam>()->Value();

            auto view = this->startCamPosition - this->startCamLookAt;
            view.Normalise();

            if (this->paramInvertZoom.Param<BoolParam>()->Value()) {
                view *= -delta;
            } else {
                view *= delta;
            }

            this->camParams->SetPosition(this->startCamPosition + view);
        }
    }
}


/*
 * megamol::vrpnModule::Manipulator::xform
 */
megamol::vrpnModule::Manipulator::QuaternionType megamol::vrpnModule::Manipulator::xform(const VectorType& u, const VectorType& v) {
    // http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    auto w = u.Cross(v);
    auto q = QuaternionType(w.X(), w.Y(), w.Z(), u.Dot(v));
    q.SetW(q.W() + q.Norm());
    q.Normalise();
    return q;
}