#include "stdafx.h"
#include "mmcore/FlagStorage.h"
#include "mmcore/FlagCall.h"

using namespace megamol;
using namespace megamol::core;


FlagStorage::FlagStorage(void)
    : getFlagsSlot("getFlags", "Provides flag data to clients.")
    , flags(std::make_shared<FlagVectorType>())
    , mut()
    , version(0) {

    this->getFlagsSlot.SetCallback(
        FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallMapFlags), &FlagStorage::mapFlagsCallback);
    this->getFlagsSlot.SetCallback(
        FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallUnmapFlags), &FlagStorage::unmapFlagsCallback);
    this->MakeSlotAvailable(&this->getFlagsSlot);
}


FlagStorage::~FlagStorage(void) { this->Release(); }


bool FlagStorage::create(void) { return true; }


void FlagStorage::release(void) {
    // intentionally empty
}


bool FlagStorage::mapFlagsCallback(core::Call& caller) {
    FlagCall* fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == nullptr) return false;

    mut.lock();
    fc->SetFlags(this->flags, this->version);

    return true;
}


bool FlagStorage::unmapFlagsCallback(core::Call& caller) {
    FlagCall* fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == nullptr) return false;

    this->flags = fc->GetFlags();
    this->version = fc->GetVersion();
    mut.unlock();

    return true;
}
