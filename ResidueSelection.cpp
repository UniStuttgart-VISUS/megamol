#include "stdafx.h"
#include "ResidueSelection.h"

using namespace megamol;
using namespace megamol::protein;


ResidueSelection::ResidueSelection(void) :
    getSelectionSlot("getSelection", "Provides selection data to clients."),
    selection() {

    this->getSelectionSlot.SetCallback( ResidueSelectionCall::ClassName(), ResidueSelectionCall::FunctionName(ResidueSelectionCall::CallForGetSelection), &ResidueSelection::getSelectionCallback);
    this->getSelectionSlot.SetCallback( ResidueSelectionCall::ClassName(), ResidueSelectionCall::FunctionName(ResidueSelectionCall::CallForSetSelection), &ResidueSelection::setSelectionCallback);
    this->MakeSlotAvailable(&this->getSelectionSlot);
}


ResidueSelection::~ResidueSelection(void){
    this->Release();
}


bool ResidueSelection::create(void) {
    return true;
}


void ResidueSelection::release(void) {
    // intentionally empty
}


bool ResidueSelection::getSelectionCallback(core::Call& caller) {
    ResidueSelectionCall *sc = dynamic_cast<ResidueSelectionCall*>(&caller);
    if (sc == NULL) return false;

    sc->SetSelectionPointer(&this->selection);

    return true;
}


bool ResidueSelection::setSelectionCallback(core::Call& caller) {
    ResidueSelectionCall *sc = dynamic_cast<ResidueSelectionCall*>(&caller);
    if (sc == NULL) return false;
    
    this->selection = *sc->GetSelectionPointer();

    return true;
}
