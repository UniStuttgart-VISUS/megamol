#include "stdafx.h"
#include "BindingSiteCall.h"

using namespace megamol;
using namespace megamol::protein;

/*
 * BindingSiteCall::BindingSiteCall
 */
const unsigned int BindingSiteCall::CallForGetData = 0;


BindingSiteCall::BindingSiteCall(void) : bindingSites(NULL),
        bindingSiteResNames(NULL) {
}


BindingSiteCall::~BindingSiteCall(void) {
    this->bindingSites = NULL;
    this->bindingSiteResNames = NULL;
}
