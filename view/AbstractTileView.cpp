/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractTileView.h"
#include "param/EnumParam.h"
#include "param/Vector2fParam.h"
#include "param/Vector4fParam.h"

using namespace megamol::core;
using vislib::graphics::CameraParameters;


/*
 * view::AbstractTileView::AbstractTileView
 */
view::AbstractTileView::AbstractTileView(void) : AbstractOverrideView(),
        eye(CameraParameters::LEFT_EYE),
        eyeSlot("eye", "The stereo projection eye"),
        projType(CameraParameters::MONO_PERSPECTIVE),
        projTypeSlot("projType", "The stereo projection type"),
        tileH(100.0f), tileSlot("tile", "The rendering tile"),
        tileW(100.0f), tileX(0.0f), tileY(0.0f), virtHeight(100.0f),
        virtSizeSlot("virtSize", "The virtual viewport size"),
        virtWidth(0.0f) {

    param::EnumParam *eyeParam = new param::EnumParam(
        static_cast<int>(CameraParameters::LEFT_EYE));
    eyeParam->SetTypePair(static_cast<int>(CameraParameters::LEFT_EYE), "Left Eye");
    eyeParam->SetTypePair(static_cast<int>(CameraParameters::RIGHT_EYE), "Right Eye");
    this->eyeSlot << eyeParam;
    this->MakeSlotAvailable(&this->eyeSlot);

    param::EnumParam *projParam = new param::EnumParam(
        static_cast<int>(CameraParameters::MONO_PERSPECTIVE));
    projParam->SetTypePair(static_cast<int>(CameraParameters::MONO_PERSPECTIVE), "Mono");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_OFF_AXIS), "Stereo OffAxis");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_PARALLEL), "Stereo Parallel");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_TOE_IN), "Stereo ToeIn");
    this->projTypeSlot << projParam;
    this->MakeSlotAvailable(&this->projTypeSlot);

    this->tileSlot << new param::Vector4fParam(vislib::math::Vector<float, 4>());
    this->MakeSlotAvailable(&this->tileSlot);

    this->virtSizeSlot << new param::Vector2fParam(vislib::math::Vector<float, 2>());
    this->MakeSlotAvailable(&this->virtSizeSlot);

}


/*
 * view::AbstractTileView::~AbstractTileView
 */
view::AbstractTileView::~AbstractTileView(void) {
    // Intentionally empty
}


/*
 * view::AbstractTileView::checkParameters
 */
void view::AbstractTileView::checkParameters(void) {
    if (this->eyeSlot.IsDirty()) {
        this->eyeSlot.ResetDirty();
        this->eye = static_cast<CameraParameters::StereoEye>(
            this->eyeSlot.Param<param::EnumParam>()->Value());
    }
    if (this->projTypeSlot.IsDirty()) {
        this->projTypeSlot.ResetDirty();
        this->projType = static_cast<CameraParameters::ProjectionType>(
            this->projTypeSlot.Param<param::EnumParam>()->Value());
    }
    if (this->tileSlot.IsDirty()) {
        this->tileSlot.ResetDirty();
        const vislib::math::Vector<float, 4> &val
            = this->tileSlot.Param<param::Vector4fParam>()->Value();
        this->tileX = val[0];
        this->tileY = val[1];
        this->tileW = val[2];
        this->tileH = val[3];
    }
    if (this->virtSizeSlot.IsDirty()) {
        this->virtSizeSlot.ResetDirty();
        const vislib::math::Vector<float, 2> &val
            = this->virtSizeSlot.Param<param::Vector2fParam>()->Value();
        this->virtWidth = val[0];
        this->virtHeight = val[1];
    }
}


/*
 * view::AbstractTileView::packMouseCoordinates
 */
void view::AbstractTileView::packMouseCoordinates(float &x, float &y) {
    x /= this->getViewportWidth();
    y /= this->getViewportHeight();
    x = this->tileX + this->tileW * x;
    y = this->tileY + this->tileH * y;
    x /= this->virtWidth;
    y /= this->virtHeight;
}
