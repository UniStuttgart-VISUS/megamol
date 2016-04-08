#include "stdafx.h"
#include "mmcore/moldyn/IntSelectionCall.h"

using namespace megamol;
using namespace megamol::core::moldyn;

/*
 * IntSelectionCall::CallForGetSelection
 */
const unsigned int IntSelectionCall::CallForGetSelection = 0;


/*
 * IntSelectionCall::CallForSetSelection
 */
const unsigned int IntSelectionCall::CallForSetSelection = 1;

/*
 *	IntSelectionCall:IntSelectionCall
 */
IntSelectionCall::IntSelectionCall(void) : selection(NULL) {
}

/*
 *	IntSelectionCall::~IntSelectionCall
 */
IntSelectionCall::~IntSelectionCall(void) {
    selection = NULL;
}
