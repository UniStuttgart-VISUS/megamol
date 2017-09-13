/*
 * AbstractView.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractView.h"
#include <climits>
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRenderView.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;
using vislib::sys::Log;







/*
 * view::AbstractView::AbstractView
 */
view::AbstractView::AbstractView(void) : Module(),
        renderSlot("render", "Connects modules requesting renderings"),
        hooks() {

    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_RENDER),
        &AbstractView::OnRenderView);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_FREEZE),
        &AbstractView::OnFreezeView);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_UNFREEZE),
        &AbstractView::OnUnfreezeView);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_SETCURSOR2DBUTTONSTATE),
        &AbstractView::onSetCursor2DButtonState);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_SETCURSOR2DPOSITION),
        &AbstractView::onSetCursor2DPosition);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_SETINPUTMODIFIER),
        &AbstractView::onSetInputModifier);
    this->renderSlot.SetCallback(view::CallRenderView::ClassName(), 
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_RESETVIEW),
        &AbstractView::onResetView);
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::AbstractView::~AbstractView
 */
view::AbstractView::~AbstractView(void) {
    this->hooks.Clear(); // DO NOT DELETE OBJECTS
}


/*
 * view::AbstractView::IsParamRelevant
 */
bool view::AbstractView::IsParamRelevant(
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    const AbstractNamedObject* ano = dynamic_cast<const AbstractNamedObject*>(this);
    if (ano == NULL) return false;
    if (param.IsNull()) return false;

    vislib::SingleLinkedList<const AbstractNamedObject*> searched;
    return ano->IsParamRelevant(searched, param);
}


/*
 * view::AbstractView::DesiredWindowPosition
 */
bool view::AbstractView::DesiredWindowPosition(int *x, int *y, int *w,
        int *h, bool *nd) {
    Module *tm = dynamic_cast<Module*>(this);
    if (tm != NULL) {

        // this is not working properly if the main module/view is placed at top namespace root
        //vislib::StringA name(tm->Name());
        //if (tm->Parent() != NULL) name = tm->Parent()->Name();
        vislib::StringA name(tm->GetDemiRootName());

        if (name.IsEmpty()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 1200,
                "View does not seem to have a name. Odd.");
        } else {
            name.Append("-Window");

            if (tm->GetCoreInstance()->Configuration().IsConfigValueSet(name)) {
                if (this->desiredWindowPosition(
                        tm->GetCoreInstance()->Configuration().ConfigValue(name),
                        x, y, w, h, nd)) {
                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 200,
                        "Loaded desired window geometry from \"%s\"", name.PeekBuffer());
                    return true;
                } else {
                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 200,
                        "Unable to load desired window geometry from \"%s\"", name.PeekBuffer());
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 1200,
                    "Unable to find window geometry settings \"%s\"", name.PeekBuffer());
            }
        }

        name = "*-Window";

        if (tm->GetCoreInstance()->Configuration().IsConfigValueSet(name)) {
            if (this->desiredWindowPosition(
                    tm->GetCoreInstance()->Configuration().ConfigValue(name),
                    x, y, w, h, nd)) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 200,
                    "Loaded desired window geometry from \"%s\"", name.PeekBuffer());
                return true;
            } else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 200,
                    "Unable to load desired window geometry from \"%s\"", name.PeekBuffer());
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 1200,
                "Unable to find window geometry settings \"%s\"", name.PeekBuffer());
        }
    }

    return false;
}


/*
 * view::AbstractView::OnRenderView
 */
bool view::AbstractView::OnRenderView(Call& call) {
    throw vislib::UnsupportedOperationException(
        "AbstractView::OnRenderView", __FILE__, __LINE__);
}


/*
 * view::AbstractView::desiredWindowPosition
 */
bool view::AbstractView::desiredWindowPosition(const vislib::StringW& str,
        int *x, int *y, int *w, int *h, bool *nd) {
    vislib::StringW v = str;
    int vi = -1;
    v.TrimSpaces();

    if (x != NULL) { *x = INT_MIN; }
    if (y != NULL) { *y = INT_MIN; }
    if (w != NULL) { *w = INT_MIN; }
    if (h != NULL) { *h = INT_MIN; }
    if (nd != NULL) { *nd = false; }

    while (!v.IsEmpty()) {
        if ((v[0] == L'X') || (v[0] == L'x')) {
            vi = 0;
        } else if ((v[0] == L'Y') || (v[0] == L'y')) {
            vi = 1;
        } else if ((v[0] == L'W') || (v[0] == L'w')) {
            vi = 2;
        } else if ((v[0] == L'H') || (v[0] == L'h')) {
            vi = 3;
        } else if ((v[0] == L'N') || (v[0] == L'n')) {
            vi = 4;
        } else if ((v[0] == L'D') || (v[0] == L'd')) {
            if (nd != NULL) {
                *nd = (vi == 4);
            }
            vi = 4;
        } else {
            Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_WARN,
                "Unexpected character %s in window position definition.\n",
                vislib::StringA(vislib::StringA(v)[0], 1).PeekBuffer());
            break;
        }
        v = v.Substring(1);
        v.TrimSpaces();

        if (vi == 4) continue; // [n]d are not followed by a number

        if (vi >= 0) {
            // now we want to parse a double :-/
            int cp = 0;
            int len = v.Length();
            while ((cp < len) && (((v[cp] >= L'0') && (v[cp] <= L'9'))
                    || (v[cp] == L'+') /*|| (v[cp] == L'.')
                    || (v[cp] == L',') */|| (v[cp] == L'-')
                    /*|| (v[cp] == L'e') || (v[cp] == L'E')*/)) {
                cp++;
            }

            try {
                int i = vislib::CharTraitsW::ParseInt(v.Substring(0, cp));
                switch (vi) {
                    case 0 :
                        if (x != NULL) { *x = i; }
                        break;
                    case 1 :
                        if (y != NULL) { *y = i; }
                        break;
                    case 2 :
                        if (w != NULL) { *w = i; }
                        break;
                    case 3 :
                        if (h != NULL) { *h = i; }
                        break;
                }
            } catch(...) {
                const char *str = "unknown";
                switch (vi) {
                    case 0 : str = "X"; break;
                    case 1 : str = "Y"; break;
                    case 2 : str = "W"; break;
                    case 3 : str = "H"; break;
                }
                vi = -1;
                Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_WARN,
                    "Unable to parse value for %s.\n", str);
            }

            v = v.Substring(cp);
        }

    }

    return true;
}


/*
 * view::AbstractView::unpackMouseCoordinates
 */
void view::AbstractView::unpackMouseCoordinates(float &x, float &y) {
    // intentionally empty
    // do something smart in the derived classes
}


/*
 * view::AbstractView::onSetCursor2DButtonState
 */
bool view::AbstractView::onSetCursor2DButtonState(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;
    this->SetCursor2DButtonState(crv->MouseButton(), crv->MouseButtonDown());
    return true;
}


/*
 * view::AbstractView::onSetCursor2DPosition
 */
bool view::AbstractView::onSetCursor2DPosition(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;

    float x = crv->MouseX();
    float y = crv->MouseY();
    this->unpackMouseCoordinates(x, y);
    this->SetCursor2DPosition(x, y);

    return true;
}


/*
 * view::AbstractView::onSetInputModifier
 */
bool view::AbstractView::onSetInputModifier(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;
    this->SetInputModifier(crv->InputModifier(), crv->MouseButtonDown());
    return true;
}


/*
 * view::AbstractView::onResetView
 */
bool view::AbstractView::onResetView(Call& call) {
    this->ResetView();
    return true;
}
