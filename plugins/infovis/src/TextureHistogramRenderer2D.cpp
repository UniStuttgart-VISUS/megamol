/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "TextureHistogramRenderer2D.h"

megamol::infovis::TextureHistogramRenderer2D::TextureHistogramRenderer2D() : BaseHistogramRenderer2D() {}

bool megamol::infovis::TextureHistogramRenderer2D::create() {
    return true;
}

void megamol::infovis::TextureHistogramRenderer2D::release() {}

bool megamol::infovis::TextureHistogramRenderer2D::GetExtents(megamol::core::view::CallRender2DGL& call) {
    return true;
}

bool megamol::infovis::TextureHistogramRenderer2D::Render(megamol::core::view::CallRender2DGL& call) {
    return true;
}
