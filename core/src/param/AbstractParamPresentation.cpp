/*
 * AbstractParamPresentation.cpp
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/AbstractParamPresentation.h"

using namespace megamol::core::param;

AbstractParamPresentation::AbstractParamPresentation(void)
    : visible(true)
    , read_only(false)
    , presentation(Presentations::Basic)
    , compatible(Presentations::Basic)
    , initialised(false) {
}

bool AbstractParamPresentation::InitPresentation(Presentations compatible, Presentations default_presentation, bool read_only, bool visible) {

    if (!this->initialised) {
        this->initialised = true;
        this->compatible = (Presentations::Basic | compatible);
        this->SetGUIVisible(visible);
        this->SetGUIReadOnly(read_only);
        return this->SetGUIPresentation(default_presentation);
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Parameter presentation can only be initilised once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}

bool AbstractParamPresentation::SetGUIPresentation(Presentations presentation) {
    if (this->IsPresentationCompatible(presentation)) {
        this->presentation = presentation;
        return true;
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Incompatible parameter presentation. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}