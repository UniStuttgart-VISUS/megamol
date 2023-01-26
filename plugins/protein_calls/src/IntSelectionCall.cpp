#include "protein_calls/IntSelectionCall.h"

using namespace megamol;
using namespace megamol::protein_calls;

/*
 * IntSelectionCall::CallForGetSelection
 */
const unsigned int IntSelectionCall::CallForGetSelection = 0;


/*
 * IntSelectionCall::CallForSetSelection
 */
const unsigned int IntSelectionCall::CallForSetSelection = 1;

/*
 * IntSelectionCall:IntSelectionCall
 */
IntSelectionCall::IntSelectionCall() : selection(NULL) {}

/*
 * IntSelectionCall::~IntSelectionCall
 */
IntSelectionCall::~IntSelectionCall() {
    selection = NULL;
}
