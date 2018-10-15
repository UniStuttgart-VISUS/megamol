/*
 * AbstractUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include <mmcore/view/AbstractInputScope.h>

namespace megamol {
namespace console {

namespace gl {
class Window;
}

class AbstractUILayer : public core::view::AbstractInputScope {
public:
    virtual bool Enabled() { return true; }

    virtual void OnResize(int w, int h) {}
    virtual void OnDraw() {}

protected:
    AbstractUILayer(gl::Window& wnd) : wnd(wnd){};
    virtual ~AbstractUILayer() = default;

    gl::Window& wnd;
};

} /* end namespace console */
} /* end namespace megamol */
