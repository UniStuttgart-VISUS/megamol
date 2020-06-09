/*
 * AbstractUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include "AbstractInputScope.h"

namespace megamol {
namespace input_events {

class AbstractUILayer : public megamol::input_events::AbstractInputScope {
public:
    virtual bool Enabled() { return true; }

    virtual void OnResize(int w, int h) {}
    virtual void OnDraw() {}

protected:
    AbstractUILayer() {}
    virtual ~AbstractUILayer() = default;
};

} /* end namespace input_events */
} /* end namespace megamol */
