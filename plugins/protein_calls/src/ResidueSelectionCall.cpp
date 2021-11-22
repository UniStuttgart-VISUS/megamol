#include "protein_calls/ResidueSelectionCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::protein_calls;

/*
 * ResidueSelectionCall::CallForGetSelection
 */
const unsigned int ResidueSelectionCall::CallForGetSelection = 0;


/*
 * ResidueSelectionCall::CallForSetSelection
 */
const unsigned int ResidueSelectionCall::CallForSetSelection = 1;

/*
 * ResidueSelectionCall::ResidueSelectionCall
 */
ResidueSelectionCall::ResidueSelectionCall(void) : selection(NULL) {}

/*
 * ResidueSelectionCall::~ResidueSelectionCall
 */
ResidueSelectionCall::~ResidueSelectionCall(void) {
    selection = NULL;
}
