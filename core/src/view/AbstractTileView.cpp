/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractTileView.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/api/MegaMolCore.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector4fParam.h"

using namespace megamol::core;


/*
 * view::AbstractTileView::AbstractTileView
 */
view::AbstractTileView::AbstractTileView(void) : AbstractOverrideView(),
        eye(thecam::Eye::left),
        eyeSlot("eye", "The stereo projection eye"),
        projType(thecam::Projection_type::perspective),
        projTypeSlot("projType", "The stereo projection type"),
        tileH(100.0f), tileSlot("tile", "The rendering tile"),
        tileW(100.0f), tileX(0.0f), tileY(0.0f), virtHeight(100.0f),
        virtSizeSlot("virtSize", "The virtual viewport size"),
        virtWidth(0.0f) {

    param::EnumParam *eyeParam = new param::EnumParam(static_cast<int>(thecam::Eye::left));
    eyeParam->SetTypePair(static_cast<int>(thecam::Eye::left), "Left Eye");
    eyeParam->SetTypePair(static_cast<int>(thecam::Eye::right), "Right Eye");
    this->eyeSlot << eyeParam;
    this->MakeSlotAvailable(&this->eyeSlot);

    param::EnumParam *projParam = new param::EnumParam(static_cast<int>(thecam::Projection_type::perspective));
    projParam->SetTypePair(static_cast<int>(thecam::Projection_type::perspective), "Mono");
    projParam->SetTypePair(static_cast<int>(thecam::Projection_type::off_axis), "Stereo OffAxis");
    projParam->SetTypePair(static_cast<int>(thecam::Projection_type::parallel), "Stereo Parallel");
    projParam->SetTypePair(static_cast<int>(thecam::Projection_type::toe_in), "Stereo ToeIn");
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


///*
// * view::AbstractTileView::AdjustTileFromContext
// */
//void view::AbstractTileView::AdjustTileFromContext(
//        const mmcRenderViewContext *context) {
////    if ((context != NULL) && (context->Window != NULL)) {
//#ifdef _WIN32
//        RECT wndRect;
//        if (::GetWindowRect(context->Window, &wndRect)) {
//            this->tileSlot.ForceSetDirty();
//            this->checkParameters();
//            this->tileH = wndRect.bottom - wndRect.top;
//            this->tileW = wndRect.right - wndRect.left;
//            this->tileX += wndRect.left;
//            this->tileY += wndRect.bottom;
//        }
//#endif /* _WIN32 */
//    }
//}


/*
 * view::AbstractTileView::initTileViewParameters
 */
void view::AbstractTileView::initTileViewParameters(void) {
    using megamol::core::utility::log::Log;
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();
    vislib::StringA v;

    v = this->getRelevantConfigValue("tveye");
    if (!v.IsEmpty()) {
        bool seteye = true;

        if (v.Equals("left", false)) {
            this->eye = thecam::Eye::left;
        } else if (v.Equals("lefteye", false)) {
            this->eye = thecam::Eye::left;
        } else if (v.Equals("left eye", false)) {
            this->eye = thecam::Eye::left;
        } else if (v.Equals("l", false)) {
            this->eye = thecam::Eye::left;
        } else if (v.Equals("right", false)) {
            this->eye = thecam::Eye::right;
        } else if (v.Equals("righteye", false)) {
            this->eye = thecam::Eye::right;
        } else if (v.Equals("right eye", false)) {
            this->eye = thecam::Eye::right;
        } else if (v.Equals("r", false)) {
            this->eye = thecam::Eye::right;
        } else {
            try {
                int ei = vislib::CharTraitsA::ParseInt(v);
                if (ei == 0) {
                    this->eye = thecam::Eye::left;
                } else if (ei == 1) {
                    this->eye = thecam::Eye::right;
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

    v = this->getRelevantConfigValue("tvproj");
    if (!v.IsEmpty()) {
        bool setproj = true;

        if (v.Equals("mono", false)) {
            this->projType = thecam::Projection_type::perspective;
        } else if (v.Equals("stereo", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereo", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereo", false)) {
            this->projType = thecam::Projection_type::parallel;
        } else if (v.Equals("stereo", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("stereooffaxis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereoparallel", false)) {
            this->projType = thecam::Projection_type::parallel;
        } else if (v.Equals("stereotoein", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("stereo off axis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereo parallel", false)) {
            this->projType = thecam::Projection_type::parallel;
        } else if (v.Equals("stereo toe in", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("off axis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("parallel", false)) {
            this->projType = thecam::Projection_type::parallel;
        } else if (v.Equals("toe in", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("offaxis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("toein", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("stereo offaxis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereo toein", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else if (v.Equals("stereooff axis", false)) {
            this->projType = thecam::Projection_type::off_axis;
        } else if (v.Equals("stereotoe in", false)) {
            this->projType = thecam::Projection_type::toe_in;
        } else {
            try {
                int pi = vislib::CharTraitsA::ParseInt(v);
                if (pi == 0) {
                    this->projType = thecam::Projection_type::perspective;
                } else if (pi == 1) {
                    this->projType = thecam::Projection_type::off_axis;
                } else if (pi == 2) {
                    this->projType = thecam::Projection_type::parallel;
                } else if (pi == 3) {
                    this->projType = thecam::Projection_type::toe_in;
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

    v = this->getRelevantConfigValue("tvview");
    if (!v.IsEmpty()) {
        try {
            if (!this->virtSizeSlot.Param<param::Vector2fParam>()->ParseValue(v)) {
                throw new vislib::Exception("ex", __FILE__, __LINE__);
            }
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse configuration value \"tvview\"\n");
        }
    }

    v = this->getRelevantConfigValue("tvtile");
    if (!v.IsEmpty()) {
        try {
            if (!this->tileSlot.Param<param::Vector4fParam>()->ParseValue(v)) {
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
        this->eye = static_cast<thecam::Eye>(
            this->eyeSlot.Param<param::EnumParam>()->Value());
    }
    if (this->projTypeSlot.IsDirty()) {
        this->projTypeSlot.ResetDirty();
        this->projType = static_cast<thecam::Projection_type>(
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
 * view::AbstractTileView::setTile
 */
bool view::AbstractTileView::setTile(const vislib::TString& val) {
    return this->tileSlot.Param<param::Vector4fParam>()->ParseValue(val);
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
