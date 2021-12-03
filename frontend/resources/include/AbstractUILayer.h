/*
 * AbstractUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include "AbstractInputScope.h"

namespace megamol {
namespace frontend_resources {

class AbstractUILayer : public megamol::frontend_resources::AbstractInputScope {
public:
    virtual bool Enabled() {
        return true;
    }

    virtual void OnResize(int w, int h) {}
    virtual void OnDraw() {}

protected:
    AbstractUILayer() {}
    virtual ~AbstractUILayer() = default;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
