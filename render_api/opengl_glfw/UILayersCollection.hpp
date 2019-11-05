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
namespace render_api {

class UILayersCollection : AbstractUILayer {
public:
    void AddUILayer(std::shared_ptr<AbstractUILayer>& uiLayer);
    void RemoveUILayer(std::shared_ptr<AbstractUILayer>& uiLayer);
    void ClearUILayers();

    bool OnKey(render_api::Key key, render_api::KeyAction action, render_api::Modifiers mods) override;
    bool OnChar(unsigned int codePoint) override;
    bool OnMouseButton(render_api::MouseButton button, render_api::MouseButtonAction action, render_api::Modifiers mods) override;
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

} // namespace render_api
} // namespace megamol