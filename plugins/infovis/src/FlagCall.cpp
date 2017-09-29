#include "stdafx.h"
#include "FlagCall.h"

using namespace megamol;
using namespace megamol::infovis;

/*
 * IntSelectionCall::CallForGetSelection
 */
const unsigned int FlagCall::CallForGetFlags= 0;


/*
 * IntSelectionCall::CallForSetSelection
 */
const unsigned int FlagCall::CallForSetFlags = 1;

/*
 *	IntSelectionCall:IntSelectionCall
 */
FlagCall::FlagCall(void) : flags(NULL) {
}

/*
 *	IntSelectionCall::~IntSelectionCall
 */
FlagCall::~FlagCall(void) {
    flags = nullptr;
}
