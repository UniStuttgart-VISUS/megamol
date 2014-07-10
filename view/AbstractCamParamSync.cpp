/*
 * AbstractCamParamSync.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractCamParamSync.h"

#include "vislib/StackTrace.h"


/*
 * megamol::core::view::AbstractCamParamSync::~AbstractCamParamSync
 */
megamol::core::view::AbstractCamParamSync::~AbstractCamParamSync(void) {
    VLAUTOSTACKTRACE;
}


/*
 * megamol::core::view::AbstractCamParamSync::AbstractCamParamSync
 */
megamol::core::view::AbstractCamParamSync::AbstractCamParamSync(void)
    : slotGetCamParams("CamParamSink", ""),
    slotSetCamParams("CamparamSource", "") {
    VLAUTOSTACKTRACE;

    this->slotGetCamParams.SetCompatibleCall<CallCamParamSyncDescription>();

    this->slotSetCamParams.SetCallback(CallCamParamSync::ClassName(),
        CallCamParamSync::FunctionName(CallCamParamSync::IDX_GET_CAM_PARAMS),
        this, &AbstractCamParamSync::onGetCamParams);
}


/*
 * megamol::core::view::AbstractCamParamSync::SyncCamParams
 */
void megamol::core::view::AbstractCamParamSync::SyncCamParams(
        CallCamParamSync::CamParams dst) {
    VLAUTOSTACKTRACE;
    auto ccp = this->slotGetCamParams.CallAs<CallCamParamSync>();
    if (ccp != nullptr) {
        (*ccp)(CallCamParamSync::IDX_GET_CAM_PARAMS);
        *dst = ccp->GetCamParams();
    }
}


/*
 * megamol::core::view::AbstractCamParamSync::onGetCamParams
 */
bool megamol::core::view::AbstractCamParamSync::onGetCamParams(Call & c) {
    VLAUTOSTACKTRACE;
    try {
        return this->OnGetCamParams(dynamic_cast<CallCamParamSync&>(c));
    } catch (...) {
        return false;
    }
}
