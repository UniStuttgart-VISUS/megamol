/*
 * UILayersCollection.hpp
 *
 * Copyright (C) 2019 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include <vector>
#include <memory>

#include "AbstractUILayer.h"

namespace megamol {
namespace console {

class UILayersCollection : AbstractUILayer {
public:
    void AddUILayer(std::shared_ptr<AbstractUILayer>& uiLayer);
    void RemoveUILayer(std::shared_ptr<AbstractUILayer>& uiLayer);
    void ClearUILayers();

    bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;
    bool OnChar(unsigned int codePoint) override;
    bool OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;
    bool OnMouseMove(double x, double y) override;
    bool OnMouseScroll(double dx, double dy) override;

    std::shared_ptr<AbstractUILayer> lastEventCaptureUILayer(); 

    void OnResize(int w, int h) override;
    void OnDraw() override;
    bool Enabled() override { return true; }
	
private:
    std::vector<std::shared_ptr<AbstractUILayer> > m_uiLayers;
    std::shared_ptr<AbstractUILayer> m_lastCapture = nullptr;
};

} // namespace console
} // namespace megamol