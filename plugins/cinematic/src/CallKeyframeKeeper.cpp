/*
* CallKeyframeKeeper.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallKeyframeKeeper.h"


using namespace megamol;
using namespace megamol::cinematic;


CallKeyframeKeeper::CallKeyframeKeeper(void) : core::AbstractGetDataCall()
    , cameraState(nullptr)
    , interpolCamPos(nullptr)
    , keyframes(nullptr)
    , interpolSteps(10)
    , selectedKeyframe()
    , dropAnimTime(0.0f)
    , dropSimTime(0.0f)
    , totalAnimTime(1.0f)
    , totalSimTime(1.0f)
    , bboxCenter(0.0f, 0.0f, 0.0f)
    , startCtrllPos(0.0f, 0.0f, 0.0f)
    , endCtrllPos(0.0f, 0.0f, 0.0f)
    , fps(24) {

}


CallKeyframeKeeper::~CallKeyframeKeeper(void) {

    this->cameraState.reset();
	this->keyframes.reset();
    this->interpolCamPos.reset();
}
