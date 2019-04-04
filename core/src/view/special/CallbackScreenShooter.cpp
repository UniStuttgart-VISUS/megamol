/*
 * CallbackScreenShooter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/job/TickCall.h"
#include "mmcore/view/special/CallbackScreenShooter.h"

#include <functional>

namespace megamol {
namespace core {
namespace view {
namespace special {

    CallbackScreenShooter::CallbackScreenShooter() : ScreenShooter(true),
        AbstractWriterParams(std::bind(&CallbackScreenShooter::MakeSlotAvailable, this, std::placeholders::_1)),
        inputSlot("inputSlot", "Slot for registering the screen shot callback function"),
        tickSlot("tickSlot", "Slot for receiving a tick") {

        // In- and output slots
        this->inputSlot.SetCompatibleCall<CallbackScreenShooterCall::CallbackScreenShooterDescription>();
        Module::MakeSlotAvailable(&this->inputSlot);

        this->tickSlot.SetCallback(job::TickCall::ClassName(), job::TickCall::FunctionName(0), &CallbackScreenShooter::Run);
        this->MakeSlotAvailable(&this->tickSlot);
    }

    CallbackScreenShooter::~CallbackScreenShooter() {
        Module::Release();
    }

    bool CallbackScreenShooter::create() {
        return true;
    }

    void CallbackScreenShooter::release() {
    }

    bool CallbackScreenShooter::Run(Call&) {
        auto* call = this->inputSlot.CallAs<CallbackScreenShooterCall>();

        if (call != nullptr)
        {
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

}
}
}
}