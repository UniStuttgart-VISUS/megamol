/*
 * Call.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/RigRendering.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
#    include "mmcore/view/Renderer2DModule.h"
#    include "mmcore/view/Renderer3DModule.h"
#    include "mmcore/view/Renderer3DModuleGL.h"
#    include "vislib/graphics/gl/IncludeAllGL.h"
#endif
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;

/*
 * Call::Call
 */
Call::Call(void) : callee(nullptr), caller(nullptr), className(nullptr), funcMap(nullptr) {
    // intentionally empty
}


/*
 * Call::~Call
 */
Call::~Call(void) {
    if (this->caller != nullptr) {
        CallerSlot* cr = this->caller;
        this->caller = nullptr; // DO NOT DELETE
        cr->ConnectCall(nullptr);
    }
    if (this->callee != nullptr) {
        this->callee->ConnectCall(nullptr);
        this->callee = nullptr; // DO NOT DELETE
    }
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 350, "destructed call \"%s\"\n", typeid(*this).name());
    ARY_SAFE_DELETE(this->funcMap);
}


/*
 * Call::operator()
 */
bool Call::operator()(unsigned int func) {
    bool res = false;
    if (this->callee != nullptr) {
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        auto f = this->callee->GetCallbackFuncName(func);
        auto parent = callee->Parent().get();
        auto p3 = dynamic_cast<core::view::Renderer3DModule*>(parent);
        auto p3_2 = dynamic_cast<core::view::Renderer3DModuleGL*>(parent);
        auto p2 = dynamic_cast<core::view::Renderer2DModule*>(parent);
        if (p3 || p3_2 || p2) {
            std::string output = dynamic_cast<core::Module*>(parent)->ClassName();
            output += "::";
            output += f;
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1234, -1, output.c_str());
            // megamol::core::utility::log::Log::DefaultLog.WriteInfo("called %s::%s", p3->ClassName(), f);
        }
#endif
        res = this->callee->InCall(this->funcMap[func], *this);
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        if (p2 || p3 || p3_2) glPopDebugGroup();
#endif
    }
    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("calling %s, idx %i, result %s (%s)", this->ClassName(), func,
    //    res ? "true" : "false", this->callee == nullptr ? "no callee" : "from callee");
    return res;
}
