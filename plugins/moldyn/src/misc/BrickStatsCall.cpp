#include "stdafx.h"
#include "moldyn/BrickStatsCall.h"

using namespace megamol;
using namespace megamol::stdplugin::moldyn;

/*
 *	IntSelectionCall:IntSelectionCall
 */
BrickStatsCall::BrickStatsCall(void) : bricks() {
}

/*
 *	IntSelectionCall::~IntSelectionCall
 */
BrickStatsCall::~BrickStatsCall(void) {
    
}


const SIZE_T BrickStatsCall::GetTypeSize() {
    return sizeof(BrickStatsType);
}