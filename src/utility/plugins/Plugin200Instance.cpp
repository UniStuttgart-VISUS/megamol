/*
 * Plugin200Instance.cpp
 * Copyright (C) 2015 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include <cassert>

using namespace megamol::core;
using namespace megamol::core::utility::plugins;


/*
 * Plugin200Instance::~Plugin200Instance
 */
Plugin200Instance::~Plugin200Instance(void) {
    // first remove the descriptions
    this->module_descriptions.Shutdown();
    this->call_descriptions.Shutdown();
    // then release the pointer to the library object. 
    assert(this->lib.use_count() > 1); // ensure the manager still holds on to
                                       // this one! Otherwise the heap we are
                                       // using will ge removed too early.
    this->lib.reset();
    // This is tricky! Derived objects are implemented in Dlls and must be
    // freed on their heap. That heap, however, will be removed when this lib
    // is unloaded.
}


/*
 * Plugin200Instance::Plugin200Instance
 */
Plugin200Instance::Plugin200Instance(const char *asm_name, const char *description)
        : AbstractPluginInstance(asm_name, description), lib() {
    // intentionally empty
}


/*
 * Plugin200Instance::store_lib
 */
void Plugin200Instance::store_lib(std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib) {
    assert(lib);
    assert(!this->lib);
    this->lib = lib;
}


/*
 * Plugin200Instance::get_lib
 */
std::shared_ptr<vislib::sys::DynamicLinkLibrary> Plugin200Instance::get_lib(void) const {
    return this->lib;
}
