#include "stdafx.h"
#include "ResidueSelection.h"

using namespace megamol;
using namespace megamol::protein;


ResidueSelection::ResidueSelection(void) :
    getSelectionSlot("getSelection", "Provides selection data to clients."),
    selection() {

	this->getSelectionSlot.SetCallback(core::moldyn::ResidueSelectionCall::ClassName(), core::moldyn::ResidueSelectionCall::FunctionName(core::moldyn::ResidueSelectionCall::CallForGetSelection), &ResidueSelection::getSelectionCallback);
	this->getSelectionSlot.SetCallback(core::moldyn::ResidueSelectionCall::ClassName(), core::moldyn::ResidueSelectionCall::FunctionName(core::moldyn::ResidueSelectionCall::CallForSetSelection), &ResidueSelection::setSelectionCallback);
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
	core::moldyn::ResidueSelectionCall *sc = dynamic_cast<core::moldyn::ResidueSelectionCall*>(&caller);
    if (sc == NULL) return false;

    sc->SetSelectionPointer(&this->selection);

    return true;
}


bool ResidueSelection::setSelectionCallback(core::Call& caller) {
	core::moldyn::ResidueSelectionCall *sc = dynamic_cast<core::moldyn::ResidueSelectionCall*>(&caller);
    if (sc == NULL) return false;
    
    this->selection = *sc->GetSelectionPointer();

    return true;
}
