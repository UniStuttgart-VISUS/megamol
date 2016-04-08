#include "stdafx.h"
#include "mmcore/moldyn/ResidueSelectionCall.h"

using namespace megamol;
using namespace megamol::core::moldyn;

/*
 * ResidueSelectionCall::CallForGetSelection
 */
const unsigned int ResidueSelectionCall::CallForGetSelection = 0;


/*
 * ResidueSelectionCall::CallForSetSelection
 */
const unsigned int ResidueSelectionCall::CallForSetSelection = 1;

/*
 *	ResidueSelectionCall::ResidueSelectionCall
 */
ResidueSelectionCall::ResidueSelectionCall(void) : selection(NULL) {
}

/*
 *	ResidueSelectionCall::~ResidueSelectionCall
 */
ResidueSelectionCall::~ResidueSelectionCall(void) {
    selection = NULL;
}
