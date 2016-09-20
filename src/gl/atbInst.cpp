/*
 * atbInst.cpp
 *
 * Copyright (C) 2008, 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/atbInst.h"
#include "AntTweakBar.h"
#include "utility/HotFixes.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::console;

namespace {

    // http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:twcopystdstringtoclientfunc
    void TW_CALL CopyStdStringToClient(
        std::string& destinationClientString,
        const std::string& sourceLibraryString) {
        // Copy the content of souceString handled by the AntTweakBar
        // library to destinationClientString handled by your application
        destinationClientString = sourceLibraryString;
    }

}

std::weak_ptr<gl::atbInst> gl::atbInst::inst;

std::shared_ptr<gl::atbInst> gl::atbInst::Instance() {
    //static std::mutex instLock;
    //std::lock_guard<std::mutex> lock(instLock);
    std::shared_ptr<gl::atbInst> i = inst.lock();
    if (!i) {
        i = std::shared_ptr<gl::atbInst>(new atbInst());
        inst = i;
    }
    return i;
}

gl::atbInst::atbInst() : error(false) {
    try {
        // Apply a global scaling factor to all fonts. This may be useful to double the size of characters on high-density display for instance.
        ::TwDefine(" GLOBAL fontscaling=1 ");
        // fontscaling must be set before calling TwInit. This is an exception.
        TwGraphAPI api = TW_OPENGL;
        if (utility::HotFixes::Instance().IsHotFixed("atbCore")) {
            api = TW_OPENGL_CORE;
        }
        if (::TwInit(api, NULL) == 0) {
            vislib::sys::Log::DefaultLog.WriteError("Failed to initialized ATB: %s", ::TwGetLastError());
            error = true;
        }
        TwDefine(" GLOBAL fontresizable=false contained=true ");
        TwCopyStdStringToClientFunc(CopyStdStringToClient);

    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to initialized ATB (exception)");
        error = true;
    }
}

gl::atbInst::~atbInst() {
    if (!error) {
        ::TwTerminate();
    }
}

#endif /* HAS_ANTTWEAKBAR */
