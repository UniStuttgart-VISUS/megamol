/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/special/CallbackScreenShooter.h"

#include <functional>

#include "mmstd/job/TickCall.h"

using namespace megamol::mmstd_gl::special;

CallbackScreenShooter::CallbackScreenShooter()
        : ScreenShooter(true)
        , AbstractWriterParams(std::bind(&CallbackScreenShooter::MakeSlotAvailable, this, std::placeholders::_1))
        , inputSlot("inputSlot", "Slot for registering the screen shot callback function")
        , tickSlot("tickSlot", "Slot for receiving a tick") {

    // In- and output slots
    this->inputSlot.SetCompatibleCall<CallbackScreenShooterCall::CallbackScreenShooterDescription>();
    Module::MakeSlotAvailable(&this->inputSlot);

    this->tickSlot.SetCallback(
        core::job::TickCall::ClassName(), core::job::TickCall::FunctionName(0), &CallbackScreenShooter::Run);
    this->MakeSlotAvailable(&this->tickSlot);
}

CallbackScreenShooter::~CallbackScreenShooter() {
    Module::Release();
}

bool CallbackScreenShooter::create() {
    return true;
}

void CallbackScreenShooter::release() {}

bool CallbackScreenShooter::Run(core::Call&) {
    auto* call = this->inputSlot.CallAs<CallbackScreenShooterCall>();

    if (call != nullptr) {
        call->SetCallback(std::bind(&CallbackScreenShooter::CreateScreenshot, this));

        return (*call)();
    }

    return true;
}

void CallbackScreenShooter::CreateScreenshot() {
    const auto filename = AbstractWriterParams::getNextFilename();

    if (filename.first) {
        ScreenShooter::createScreenshot(filename.second);
    }
}
