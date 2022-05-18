#include "IntSelection.h"
#include "protein_calls/IntSelectionCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::protein;


IntSelection::IntSelection(void)
        : getSelectionSlot("getSelection", "Provides selection data to clients.")
        , selection() {

    this->getSelectionSlot.SetCallback(protein_calls::IntSelectionCall::ClassName(),
        protein_calls::IntSelectionCall::FunctionName(protein_calls::IntSelectionCall::CallForGetSelection),
        &IntSelection::getSelectionCallback);
    this->getSelectionSlot.SetCallback(protein_calls::IntSelectionCall::ClassName(),
        protein_calls::IntSelectionCall::FunctionName(protein_calls::IntSelectionCall::CallForSetSelection),
        &IntSelection::setSelectionCallback);
    this->MakeSlotAvailable(&this->getSelectionSlot);
}


IntSelection::~IntSelection(void) {
    this->Release();
}


bool IntSelection::create(void) {
    return true;
}


void IntSelection::release(void) {
    // intentionally empty
}


bool IntSelection::getSelectionCallback(core::Call& caller) {
    protein_calls::IntSelectionCall* sc = dynamic_cast<protein_calls::IntSelectionCall*>(&caller);
    if (sc == NULL)
        return false;

    sc->SetSelectionPointer(&this->selection);

    return true;
}


bool IntSelection::setSelectionCallback(core::Call& caller) {
    protein_calls::IntSelectionCall* sc = dynamic_cast<protein_calls::IntSelectionCall*>(&caller);
    if (sc == NULL)
        return false;

    this->selection = *sc->GetSelectionPointer();

    return true;
}
