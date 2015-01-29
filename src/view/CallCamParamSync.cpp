#include "stdafx.h"
#include "view/CallCamParamSync.h"


/*
 * megamol::pcl::CallPcd::FunctionCount
 */
unsigned int megamol::core::view::CallCamParamSync::FunctionCount(void) {
    VLAUTOSTACKTRACE;
    return (sizeof(CallCamParamSync::INTENTS)
        / sizeof(*CallCamParamSync::INTENTS));
}


/*
 * megamol::pcl::CallPcd::FunctionName
 */
const char *megamol::core::view::CallCamParamSync::FunctionName(
        unsigned int idx) {
    VLAUTOSTACKTRACE;
    if (idx < CallCamParamSync::FunctionCount()) {
        return CallCamParamSync::INTENTS[idx];
    } else {
        return "";
    }
}


/*
 * megamol::core::misc::VolumetricDataCall::IDX_GET_CAM_PARAMS
 */
const unsigned int megamol::core::view::CallCamParamSync::IDX_GET_CAM_PARAMS = 0;


/*
 * megamol::core::view::CallCamParamSync::CallCamParamSync
 */
megamol::core::view::CallCamParamSync::CallCamParamSync(void) {
    VLAUTOSTACKTRACE;
}


/*
 * megamol::core::view::CallCamParamSync::~CallCamParamSync
 */
megamol::core::view::CallCamParamSync::~CallCamParamSync(void) {
    VLAUTOSTACKTRACE;
}


/*
 * megamol::core::view::CallCamParamSync::INTENTS
 */
const char *megamol::core::view::CallCamParamSync::INTENTS[] = {
    "getCamParams"
};
