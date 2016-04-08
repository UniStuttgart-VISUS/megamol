#include "stdafx.h"
#include "IntSelection.h"
#include "mmcore/moldyn/IntSelectionCall.h"

using namespace megamol;
using namespace megamol::protein;


IntSelection::IntSelection(void) :
    getSelectionSlot("getSelection", "Provides selection data to clients."),
    selection() {

	this->getSelectionSlot.SetCallback(core::moldyn::IntSelectionCall::ClassName(), core::moldyn::IntSelectionCall::FunctionName(core::moldyn::IntSelectionCall::CallForGetSelection), &IntSelection::getSelectionCallback);
	this->getSelectionSlot.SetCallback(core::moldyn::IntSelectionCall::ClassName(), core::moldyn::IntSelectionCall::FunctionName(core::moldyn::IntSelectionCall::CallForSetSelection), &IntSelection::setSelectionCallback);
    this->MakeSlotAvailable(&this->getSelectionSlot);
}


IntSelection::~IntSelection(void){
    this->Release();
}


bool IntSelection::create(void) {
    return true;
}


void IntSelection::release(void) {
    // intentionally empty
}


bool IntSelection::getSelectionCallback(core::Call& caller) {
	core::moldyn::IntSelectionCall *sc = dynamic_cast<core::moldyn::IntSelectionCall*>(&caller);
    if (sc == NULL) return false;

    sc->SetSelectionPointer(&this->selection);

    return true;
}


bool IntSelection::setSelectionCallback(core::Call& caller) {
	core::moldyn::IntSelectionCall *sc = dynamic_cast<core::moldyn::IntSelectionCall*>(&caller);
    if (sc == NULL) return false;
    
    this->selection = *sc->GetSelectionPointer();

    return true;
}
