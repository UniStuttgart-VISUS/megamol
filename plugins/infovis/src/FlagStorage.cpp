#include "stdafx.h"
#include "FlagStorage.h"
#include "FlagCall.h"

using namespace megamol;
using namespace megamol::infovis;


FlagStorage::FlagStorage(void) :
    getFlagsSlot("getFlags", "Provides flag data to clients."),
    flags(), crit() {

    this->getFlagsSlot.SetCallback(FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallForGetFlags), &FlagStorage::getFlagsCallback);
    this->getFlagsSlot.SetCallback(FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallForSetFlags), &FlagStorage::setFlagsCallback);
    this->MakeSlotAvailable(&this->getFlagsSlot);
}


FlagStorage::~FlagStorage(void) {
    this->Release();
}


bool FlagStorage::create(void) {
    return true;
}


void FlagStorage::release(void) {
    // intentionally empty
}


bool FlagStorage::getFlagsCallback(core::Call& caller) {
    FlagCall *fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == NULL) return false;

    crit.Lock();
    //fc->SetFlags(this->flags);
    // TODO less yucky
    fc->flags = this->flags;
    crit.Unlock();

    return true;
}


bool FlagStorage::setFlagsCallback(core::Call& caller) {
    FlagCall *fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == NULL) return false;
    
    crit.Lock();
    //this->flags = fc->GetFlags();
    // TODO less yucky
    this->flags = fc->flags;
    crit.Unlock();

    return true;
}
