/*
 * glfwInst.cpp
 *
 * Copyright (C) 2008, 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#ifndef USE_EGL

#include "gl/glfwInst.h"
#include "GLFW/glfw3.h"
#include "vislib/sys/Log.h"
//#include <mutex>

using namespace megamol;
using namespace megamol::console;

std::weak_ptr<gl::glfwInst> gl::glfwInst::inst;

std::shared_ptr<gl::glfwInst> gl::glfwInst::Instance() {
    //static std::mutex instLock;
    //std::lock_guard<std::mutex> lock(instLock);
    std::shared_ptr<gl::glfwInst> i = inst.lock();
    if (!i) {
        i = std::shared_ptr<gl::glfwInst>(new glfwInst());
        inst = i;
    }
    return i;
}

gl::glfwInst::glfwInst() : error(false) {
    try {
        if (::glfwInit() != GLFW_TRUE) {
            vislib::sys::Log::DefaultLog.WriteError("glfwInit failed");
            error = true;
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("glfwInit failed (exception)");
        error = true;
    }
}

gl::glfwInst::~glfwInst() {
    if (!error) {
        ::glfwTerminate();
    }
}

#endif // USE_EGL
