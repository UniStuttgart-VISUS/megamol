#include "stdafx.h"
#include "IntSelection.h"
#include "IntSelectionCall.h"

using namespace megamol;
using namespace megamol::protein;


IntSelection::IntSelection(void) :
    getSelectionSlot("getSelection", "Provides selection data to clients."),
    selection() {

    this->getSelectionSlot.SetCallback( IntSelectionCall::ClassName(), IntSelectionCall::FunctionName(IntSelectionCall::CallForGetSelection), &IntSelection::getSelectionCallback);
    this->getSelectionSlot.SetCallback( IntSelectionCall::ClassName(), IntSelectionCall::FunctionName(IntSelectionCall::CallForSetSelection), &IntSelection::setSelectionCallback);
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
    IntSelectionCall *sc = dynamic_cast<IntSelectionCall*>(&caller);
    if (sc == NULL) return false;

    sc->SetSelectionPointer(&this->selection);

    return true;
}


bool IntSelection::setSelectionCallback(core::Call& caller) {
    IntSelectionCall *sc = dynamic_cast<IntSelectionCall*>(&caller);
    if (sc == NULL) return false;
    
    this->selection = *sc->GetSelectionPointer();

    return true;
}
