/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_INFOVIS_TEXTUREHISTOGRAMRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_TEXTUREHISTOGRAMRENDERER2D_H_INCLUDED

#include "BaseHistogramRenderer2D.h"

namespace megamol::infovis {

class TextureHistogramRenderer2D : public BaseHistogramRenderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TextureHistogramRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Histogram renderer for textures.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TextureHistogramRenderer2D();

protected:
    bool create() override;

    void release() override;

    bool GetExtents(core::view::CallRender2DGL& call) override;

    bool Render(core::view::CallRender2DGL& call) override;
};

} // namespace megamol::infovis

#endif // MEGAMOL_INFOVIS_TEXTUREHISTOGRAMRENDERER2D_H_INCLUDED
