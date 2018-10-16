#include "stdafx.h"
#include "FlagStorage.h"
#include "infovis/FlagCall.h"
#include "mmcore/CallerSlot.h"

using namespace megamol;
using namespace megamol::infovis;

/*
 * FlagStorage::FlagStorage
 */
FlagStorage::FlagStorage(void) : getFlagsSlot("getFlags", "Provides flag data to clients."), flags(), crit() {

    this->getFlagsSlot.SetCallback(
        FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallForGetFlags), &FlagStorage::getFlagsCallback);
    this->getFlagsSlot.SetCallback(
        FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallForSetFlags), &FlagStorage::setFlagsCallback);
    this->getFlagsSlot.SetCallback(
        FlagCall::ClassName(), FlagCall::FunctionName(FlagCall::CallForResetDirty), &FlagStorage::resetDirtyCallback);
    this->MakeSlotAvailable(&this->getFlagsSlot);
}

/*
 * FlagStorage::FlagStorage
 */
FlagStorage::~FlagStorage(void) { this->Release(); }

/*
 * FlagStorage::create
 */
bool FlagStorage::create(void) { return true; }

/*
 * FlagStorage::release
 */
void FlagStorage::release(void) {
    // intentionally empty
}

/*
 * FlagStorage::getFlagsCallback
 */
bool FlagStorage::getFlagsCallback(core::Call& caller) {
    FlagCall* fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == nullptr) return false;

    crit.Lock();
    // fc->SetFlags(this->flags);
    // TODO less yucky
    fc->flags = this->flags;
    crit.Unlock();

    return true;
}

/*
 * FlagStorage::setFlagsCallback
 */
bool FlagStorage::setFlagsCallback(core::Call& caller) {
    FlagCall* fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == nullptr) return false;

    crit.Lock();
    // this->flags = fc->GetFlags();
    // TODO less yucky
    this->flags = fc->flags;
    crit.Unlock();

    return true;
}

/*
 * FlagStorage::resetDirtyCallback
 */
bool FlagStorage::resetDirtyCallback(core::Call& caller) {
    FlagCall* fc = dynamic_cast<FlagCall*>(&caller);
    if (fc == nullptr) return false;

    auto cl = fc->PeekCallerSlot();
    uintptr_t val = reinterpret_cast<uintptr_t>(cl);

    this->dirtyFlags[val] = false;

    return true;
}
