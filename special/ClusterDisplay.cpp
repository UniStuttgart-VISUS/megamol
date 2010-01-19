/*
 * ClusterDisplay.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterDisplay.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "special/ClusterDisplayTile.h"
#include "special/ClusterDisplayPlane.h"
#include "view/CallCursorInput.h"
#include "vislib/Array.h"
#include "vislib/mathfunctions.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"

using namespace megamol::core;
using vislib::sys::Log;


/*
 * special::ClusterDisplay::ClusterDisplay
 */
special::ClusterDisplay::ClusterDisplay(void) : special::RenderSlave(),
        viewplane("viewplane", "The id of the view plane"),
        viewTile("viewtile", "The tile rectangle on the view plane"),
        cursorInputSlot("cursorInput", "Slot for sending the cursor input") {
    // We do not make the slot 'AbstractView::getRenderViewSlot()' available,
    // since we are a top level view and do not want to be rendered inside
    // another one

    this->viewplane << new param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->viewplane);

    this->viewTile << new param::StringParam("0;0;1;1");
    this->MakeSlotAvailable(&this->viewTile);

    this->cursorInputSlot.SetCompatibleCall<view::CallCursorInputDescription>();
    this->MakeSlotAvailable(&this->cursorInputSlot);

}


/*
 * special::ClusterDisplay::~ClusterDisplay
 */
special::ClusterDisplay::~ClusterDisplay(void) {
    this->Release();
}


/*
 * special::ClusterDisplay::Render
 */
void special::ClusterDisplay::Render(void) {

    if (this->viewplane.IsDirty() || this->viewTile.IsDirty()) {
        this->viewplane.ResetDirty();
        this->viewTile.ResetDirty();

        ClusterDisplayTile tile; // tile.plane = 0;
        const ClusterDisplayPlane *plane = NULL;

        vislib::Array<vislib::TString> el = vislib::TStringTokeniser::Split(
            this->viewTile.Param<param::StringParam>()->Value(), _T(";"));
        if (el.Count() == 4) {
            try {
                tile.SetX(float(vislib::TCharTraits::ParseDouble(el[0])));
                tile.SetY(float(vislib::TCharTraits::ParseDouble(el[1])));
                tile.SetWidth(float(vislib::TCharTraits::ParseDouble(el[2])));
                tile.SetHeight(float(vislib::TCharTraits::ParseDouble(el[3])));
                tile.SetPlane(this->viewplane.Param<param::IntParam>()->Value());
            } catch(...) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to parse view tile parameter: float parse error");
            }
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to parse view tile parameter: Syntax error");
        }

        if (tile.Plane() == 0) {
            tile.SetPlane(this->viewplane.Param<param::IntParam>()->Value());
            plane = ClusterDisplayPlane::Plane(tile.Plane(),
                *this->GetCoreInstance());
            if (plane != NULL) {
                tile.SetX(0.0f);
                tile.SetY(0.0f);
                tile.SetWidth(plane->Width());
                tile.SetHeight(plane->Height());
            }
        } else {
            plane = ClusterDisplayPlane::Plane(tile.Plane(),
                *this->GetCoreInstance());
        }

        if (plane != NULL) {
            this->setClusterDisplayTile(*plane, tile);
        } else {
            this->resetClusterDisplayTile();
        }
    }

    RenderSlave::Render();

}


/*
 * special::ClusterDisplay::ResetView
 */
void special::ClusterDisplay::ResetView(void) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) (*cci)(3);
}


/*
 * special::ClusterDisplay::SetCursor2DButtonState
 */
void special::ClusterDisplay::SetCursor2DButtonState(unsigned int btn,
        bool down) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        cci->Btn() = btn;
        cci->Down() = down;
        (*cci)(0);
    }
}


/*
 * special::ClusterDisplay::SetCursor2DPosition
 */
void special::ClusterDisplay::SetCursor2DPosition(float x, float y) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        float w, h, ox, oy, s;
        if (this->displayPlane().Type() == ClusterDisplayPlane::TYPE_VOID) {
            w = static_cast<float>(this->viewWidth());
            h = static_cast<float>(this->viewHeight());
            ox = 0.0f;
            oy = 0.0;
        } else {
            w = this->displayPlane().Width();
            h = this->displayPlane().Height();
            ox = this->displayTile().X();
            oy = this->displayTile().Y();
        }
        s = vislib::math::Min(w, h);

        cci->X() = (x + ox - 0.5f * w) / s;
        cci->Y() = (y + oy - 0.5f * h) / s;

        (*cci)(1);
    }
}


/*
 * special::ClusterDisplay::SetInputModifier
 */
void special::ClusterDisplay::SetInputModifier(mmcInputModifier mod,
        bool down) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        cci->Mod() = mod;
        cci->Down() = down;
        (*cci)(2);
    }
}


/*
 * special::ClusterDisplay::DesiredWindowPosition
 */
bool special::ClusterDisplay::DesiredWindowPosition(int *x, int *y, int *w,
        int *h, bool *nd) {
    vislib::StringA name;

    const utility::Configuration &cfg
        = this->GetCoreInstance()->Configuration();

    name.Format("%s-%s", this->instName().PeekBuffer(), "Window");
    if (cfg.IsConfigValueSet(name)) {
        if (this->desiredWindowPosition(cfg.ConfigValue(name),
                x, y, w, h, nd)) {
            return true;
        }
    }
    return view::AbstractView::DesiredWindowPosition(x, y, w, h, nd);
}


/*
 * special::ClusterDisplay::create
 */
bool special::ClusterDisplay::create(void) {
    // parent is not yet set, so we do not know the instance name
    return RenderSlave::create();
}


/*
 * special::ClusterDisplay::release
 */
void special::ClusterDisplay::release(void) {
    RenderSlave::release();
}


/*
 * special::ClusterDisplay::instName
 */
const vislib::StringA& special::ClusterDisplay::instName(void) {
    if (this->Parent() != NULL) {
        return this->Parent()->Name();
    }
    return this->Name();
}
