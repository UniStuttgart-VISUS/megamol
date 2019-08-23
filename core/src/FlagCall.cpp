#include "stdafx.h"
#include "mmcore/FlagCall.h"

using namespace megamol;
using namespace megamol::core;

/*
 * IntSelectionCall::CallForGetSelection
 */
const unsigned int FlagCall::CallMapFlags = 0;


/*
 * IntSelectionCall::CallForSetSelection
 */
const unsigned int FlagCall::CallUnmapFlags = 1;

/*
 *	IntSelectionCall:IntSelectionCall
 */
FlagCall::FlagCall(void) : flags() {}

/*
 *	IntSelectionCall::~IntSelectionCall
 */
FlagCall::~FlagCall(void) { flags = nullptr; }
