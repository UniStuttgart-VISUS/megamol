/*
 * ApiHandle.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "param/ParamHandle.h"


/*
 * megamol::core::param::ParamHandle::ParamHandle
 */
megamol::core::param::ParamHandle::ParamHandle(
        const megamol::core::CoreInstance& inst,
        const vislib::SmartPtr<megamol::core::param::AbstractParam>& param)
        : inst(inst), param(param) {
    // intentionally empty
}


/*
 * megamol::core::param::ParamHandle::ParamHandle
 */
megamol::core::param::ParamHandle::ParamHandle(
        const megamol::core::param::ParamHandle& src) : inst(src.inst),
        param(src.param) {
    // intentionally empty
}


/*
 * megamol::core::param::ParamHandle::~ParamHandle
 */
megamol::core::param::ParamHandle::~ParamHandle(void) {
    this->param = NULL; // paranoia
}


/*
 * megamol::core::param::ParamHandle::GetIDString
 */
void megamol::core::param::ParamHandle::GetIDString(vislib::StringA& outID) {
    try {
        outID = inst.FindParameterName(this->param);
    } catch(...) {
        outID.Clear();
    }
}
