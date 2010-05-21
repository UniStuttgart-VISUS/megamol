/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractTileView.h"
#include "CoreInstance.h"
#include "utility/Configuration.h"
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
 * view::AbstractTileView::initTileViewParameters
 */
void view::AbstractTileView::initTileViewParameters(void) {
    using vislib::sys::Log;
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("tveye")) {
        vislib::StringA v(cfg.ConfigValue("tveye"));
        bool seteye = true;

        if (v.Equals("left", false)) {
            this->eye = CameraParameters::LEFT_EYE;
        } else if (v.Equals("lefteye", false)) {
            this->eye = CameraParameters::LEFT_EYE;
        } else if (v.Equals("left eye", false)) {
            this->eye = CameraParameters::LEFT_EYE;
        } else if (v.Equals("l", false)) {
            this->eye = CameraParameters::LEFT_EYE;
        } else if (v.Equals("right", false)) {
            this->eye = CameraParameters::RIGHT_EYE;
        } else if (v.Equals("righteye", false)) {
            this->eye = CameraParameters::RIGHT_EYE;
        } else if (v.Equals("right eye", false)) {
            this->eye = CameraParameters::RIGHT_EYE;
        } else if (v.Equals("r", false)) {
            this->eye = CameraParameters::RIGHT_EYE;
        } else {
            try {
                int ei = vislib::CharTraitsA::ParseInt(v);
                if (ei == 0) {
                    this->eye = CameraParameters::LEFT_EYE;
                } else if (ei == 1) {
                    this->eye = CameraParameters::RIGHT_EYE;
                } else {
                    seteye = false;
                }
            } catch(...) {
                seteye = false;
            }
        }
        if (seteye) {
            this->eyeSlot.Param<param::EnumParam>()->SetValue(static_cast<int>(this->eye));
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse configuration value \"tveye\"\n");
        }
    }

    if (cfg.IsConfigValueSet("tvproj")) {
        vislib::StringA v(cfg.ConfigValue("tvproj"));
        bool setproj = true;

        if (v.Equals("mono", false)) {
            this->projType = CameraParameters::MONO_PERSPECTIVE;
        } else if (v.Equals("stereo", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereo", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereo", false)) {
            this->projType = CameraParameters::STEREO_PARALLEL;
        } else if (v.Equals("stereo", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("stereooffaxis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereoparallel", false)) {
            this->projType = CameraParameters::STEREO_PARALLEL;
        } else if (v.Equals("stereotoein", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("stereo off axis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereo parallel", false)) {
            this->projType = CameraParameters::STEREO_PARALLEL;
        } else if (v.Equals("stereo toe in", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("off axis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("parallel", false)) {
            this->projType = CameraParameters::STEREO_PARALLEL;
        } else if (v.Equals("toe in", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("offaxis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("toein", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("stereo offaxis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereo toein", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else if (v.Equals("stereooff axis", false)) {
            this->projType = CameraParameters::STEREO_OFF_AXIS;
        } else if (v.Equals("stereotoe in", false)) {
            this->projType = CameraParameters::STEREO_TOE_IN;
        } else {
            try {
                int pi = vislib::CharTraitsA::ParseInt(v);
                if (pi == 0) {
                    this->projType = CameraParameters::MONO_PERSPECTIVE;
                } else if (pi == 1) {
                    this->projType = CameraParameters::STEREO_OFF_AXIS;
                } else if (pi == 2) {
                    this->projType = CameraParameters::STEREO_PARALLEL;
                } else if (pi == 3) {
                    this->projType = CameraParameters::STEREO_TOE_IN;
                } else {
                    setproj = false;
                }
            } catch(...) {
                setproj = false;
            }
        }

        if (setproj) {
            this->projTypeSlot.Param<param::EnumParam>()->SetValue(static_cast<int>(this->projType));
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse configuration value \"tvproj\"\n");
        }
    }

    if (cfg.IsConfigValueSet("tvview")) {
        try {
            if (!this->virtSizeSlot.Param<param::Vector2fParam>()->ParseValue(cfg.ConfigValue("tvview"))) {
                throw new vislib::Exception("ex", __FILE__, __LINE__);
            }
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse configuration value \"tvview\"\n");
        }
    }

    if (cfg.IsConfigValueSet("tvtile")) {
        try {
            if (!this->tileSlot.Param<param::Vector4fParam>()->ParseValue(cfg.ConfigValue("tvtile"))) {
                throw new vislib::Exception("ex", __FILE__, __LINE__);
            }
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse configuration value \"tvtile\"\n");
        }
    }

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
