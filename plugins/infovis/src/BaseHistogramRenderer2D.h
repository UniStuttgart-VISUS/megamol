/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED

#include "Renderer2D.h"

namespace megamol::infovis {

class BaseHistogramRenderer2D : public Renderer2D {
public:
    BaseHistogramRenderer2D();

    ~BaseHistogramRenderer2D() override = default;
};

} // namespace megamol::infovis

#endif // MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED
