#include "stdafx.h"
#include "IntSelectionCall.h"

using namespace megamol;
using namespace megamol::protein_cuda;

/*
 * IntSelectionCall::CallForGetSelection
 */
const unsigned int IntSelectionCall::CallForGetSelection = 0;


/*
 * IntSelectionCall::CallForSetSelection
 */
const unsigned int IntSelectionCall::CallForSetSelection = 1;

IntSelectionCall::IntSelectionCall(void) : selection(NULL) {
}


IntSelectionCall::~IntSelectionCall(void) {
    selection = NULL;
}
