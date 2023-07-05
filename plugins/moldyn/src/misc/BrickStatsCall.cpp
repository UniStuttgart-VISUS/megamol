#include "moldyn/BrickStatsCall.h"

using namespace megamol;
using namespace megamol::moldyn;

/*
 * IntSelectionCall:IntSelectionCall
 */
BrickStatsCall::BrickStatsCall() : bricks() {}

/*
 * IntSelectionCall::~IntSelectionCall
 */
BrickStatsCall::~BrickStatsCall() {}


const SIZE_T BrickStatsCall::GetTypeSize() {
    return sizeof(BrickStatsType);
}
