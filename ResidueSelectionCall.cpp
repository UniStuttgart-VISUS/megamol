#include "stdafx.h"
#include "ResidueSelectionCall.h"

using namespace megamol;
using namespace megamol::protein;

/*
 * ResidueSelectionCall::CallForGetSelection
 */
const unsigned int ResidueSelectionCall::CallForGetSelection = 0;


/*
 * ResidueSelectionCall::CallForSetSelection
 */
const unsigned int ResidueSelectionCall::CallForSetSelection = 1;

ResidueSelectionCall::ResidueSelectionCall(void) : selection(NULL) {
}


ResidueSelectionCall::~ResidueSelectionCall(void) {
    selection = NULL;
}
