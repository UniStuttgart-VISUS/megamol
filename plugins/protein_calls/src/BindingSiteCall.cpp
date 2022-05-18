#include "protein_calls/BindingSiteCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;

/*
 * BindingSiteCall::BindingSiteCall
 */
const unsigned int BindingSiteCall::CallForGetData = 0;


BindingSiteCall::BindingSiteCall(void)
        : megamol::core::Call()
        , bindingSites(NULL)
        , bindingSiteResNames(NULL)
        , bindingSiteNames(NULL)
        , bindingSiteDescriptions(NULL)
        , bindingSiteColors(NULL)
        , enzymeCase(false)
        , isGxType(true) {}


BindingSiteCall::~BindingSiteCall(void) {
    this->bindingSites = NULL;
    this->bindingSiteResNames = NULL;
    this->bindingSiteNames = NULL;
    this->bindingSiteDescriptions = NULL;
    this->bindingSiteColors = NULL;
}
