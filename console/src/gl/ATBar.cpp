/*
 * gl/ATBar.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/ATBar.h"

megamol::console::gl::ATBar::ATBar(const char* name) : atb(), barName(name), bar(nullptr) {
    atb = atbInst::Instance();
    bar = ::TwNewBar(name);
}

megamol::console::gl::ATBar::~ATBar() {
    if (bar != nullptr) {
        ::TwDeleteBar(bar);
        bar = nullptr;
    }
}
#endif
