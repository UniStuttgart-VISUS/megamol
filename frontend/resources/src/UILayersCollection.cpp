/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "UILayersCollection.hpp"

#include <algorithm>

/// #include "mmcore/utility/log/Log.h"

#define init          \
    bool ret = false; \
    m_lastCapture = nullptr;
#define abort                \
    {                        \
        m_lastCapture = uil; \
        break;               \
    }

using namespace megamol::frontend_resources;

std::shared_ptr<AbstractUILayer> UILayersCollection::lastEventCaptureUILayer() {
    return std::move(m_lastCapture);
}

void UILayersCollection::AddUILayer(std::shared_ptr<AbstractUILayer>& uiLayer) {
    auto it = std::find(m_uiLayers.begin(), m_uiLayers.end(), uiLayer);
    if (it != m_uiLayers.end()) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("uiLayer already part of the collection");
        return;
    }
    m_uiLayers.push_back(uiLayer);
}
void UILayersCollection::RemoveUILayer(std::shared_ptr<AbstractUILayer>& uiLayer) {
    auto it = std::find(m_uiLayers.begin(), m_uiLayers.end(), uiLayer);
    if (it == m_uiLayers.end())
        return;
    m_uiLayers.erase(it);
}
void UILayersCollection::ClearUILayers() {
    m_lastCapture = nullptr;
    m_uiLayers.clear();
}

bool UILayersCollection::OnKey(Key key, KeyAction action, Modifiers mods) {
    init;
    for (auto& uil : m_uiLayers) {
        if (!uil->Enabled())
            continue;
        if (ret = uil->OnKey(key, action, mods))
            abort;
    }
    return ret;
}

bool UILayersCollection::OnChar(unsigned int codePoint) {
    init;
    for (auto& uil : m_uiLayers) {
        if (!uil->Enabled())
            continue;
        if (ret = uil->OnChar(codePoint))
            abort;
    }
    return ret;
}

bool UILayersCollection::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    init;
    for (auto& uil : m_uiLayers) {
        if (!uil->Enabled())
            continue;
        if (ret = uil->OnMouseButton(button, action, mods))
            abort;
    }
    return ret;
}

bool UILayersCollection::OnMouseMove(double x, double y) {
    init;
    for (auto& uil : m_uiLayers) {
        if (!uil->Enabled())
            continue;
        if (ret = uil->OnMouseMove(x, y))
            abort;
    }
    return ret;
}

bool UILayersCollection::OnMouseScroll(double dx, double dy) {
    init;
    for (auto& uil : m_uiLayers) {
        if (!uil->Enabled())
            continue;
        if (ret = uil->OnMouseScroll(dx, dy))
            abort;
    }
    return ret;
}

void UILayersCollection::OnResize(int w, int h) {
    if ((w > 0) && (h > 0)) {
        for (auto& uil : m_uiLayers) {
            // we inform even disabled layers, since we would need to know and update as soon as they get enabled.
            uil->OnResize(w, h);
        }
    }
}

void UILayersCollection::OnDraw() {
    for (auto& uil : this->m_uiLayers) {
        if (!uil->Enabled())
            continue;
        uil->OnDraw();
    }
}

//bool UILayersCollection::Enabled() {
//    return true;
//}
