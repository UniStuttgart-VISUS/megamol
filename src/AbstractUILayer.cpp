/*
 * AbstractUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "AbstractUILayer.h"
#include "gl/Window.h"

using namespace megamol;
using namespace megamol::console;

bool AbstractUILayer::Enabled() {
    return true;
}

void AbstractUILayer::onResize(int w, int h) { }
void AbstractUILayer::onDraw() { }
bool AbstractUILayer::onKey(Key key, int scancode, KeyAction action, Modifiers mods) { return false; }
bool AbstractUILayer::onChar(unsigned int charcode) { return false; }
bool AbstractUILayer::onMouseMove(double x, double y) { return false; }
bool AbstractUILayer::onMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) { return false; }
bool AbstractUILayer::onMouseWheel(double x, double y) { return false; }

AbstractUILayer::AbstractUILayer(gl::Window& wnd) : wnd(wnd) {
}

AbstractUILayer::~AbstractUILayer() {
}
