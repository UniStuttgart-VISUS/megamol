/*
 * ViewUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "ViewUILayer.h"
#include "mmcore/api/MegaMolCore.h"

using namespace megamol;
using namespace megamol::console;

ViewUILayer::ViewUILayer(void* viewHandle) : hView(viewHandle) {}

ViewUILayer::~ViewUILayer() {
    hView = nullptr; // handle memory is owned by Window and will be deleted there
}

void ViewUILayer::OnResize(int w, int h) {
    ::mmcResizeView(hView, static_cast<unsigned int>(w), static_cast<unsigned int>(h));
}

bool ViewUILayer::OnKey(Key key, KeyAction action, Modifiers mods) {
    return ::mmcSendKeyEvent(hView,
		static_cast<mmcInputKey>(key),
		static_cast<mmcInputKeyAction>(action),
		static_cast<mmcInputModifiers>(mods.toInt()));
}

bool ViewUILayer::OnChar(unsigned int codePoint) {
    return ::mmcSendCharEvent(hView, codePoint);
}

bool ViewUILayer::OnMouseButton(
    MouseButton button, MouseButtonAction action, Modifiers mods) {
    return ::mmcSendMouseButtonEvent(hView, 
		static_cast<mmcInputButton>(button), 
		static_cast<mmcInputButtonAction>(action),
        static_cast<mmcInputModifiers>(mods.toInt()));
}

bool ViewUILayer::OnMouseMove(double x, double y) {
    return ::mmcSendMouseMoveEvent(hView, static_cast<float>(x), static_cast<float>(y));
}

bool ViewUILayer::OnMouseScroll(double dx, double dy) {
    return ::mmcSendMouseScrollEvent(hView, static_cast<float>(dx), static_cast<float>(dy));
}
