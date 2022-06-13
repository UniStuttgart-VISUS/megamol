/*
 * OverlayRenderer.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CINEMATIC_OVERLAYRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATIC_OVERLAYRENDERER_H_INCLUDED
#pragma once


#include "mmcore/view/AbstractView.h"
#include "mmcore_gl/utility/RenderUtils.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <iomanip>


namespace megamol {
namespace cinematic_gl {


/** ************************************************************************
 * Renders various kinds of overlays
 */
class OverlayRenderer : public megamol::core::view::RendererModule<mmstd_gl::CallRender3DGL, mmstd_gl::ModuleGL>,
                        megamol::core_gl::utility::RenderUtils {
public:
    virtual std::vector<std::string> requested_lifetime_resources() {
        return {"MegaMolGraph"};
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OverlayRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renders various kinds of overlays.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    OverlayRenderer();

    /** Dtor. */
    virtual ~OverlayRenderer();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create();

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

private:
    struct Rectangle {
        float left;
        float right;
        float top;
        float bottom;
    };

    typedef megamol::core::utility::SDFFont::Alignment Anchor;

    enum Mode { TEXTURE, TRANSPORT_CTRL, PARAMETER, LABEL };

    // Explicit numbering required as indices in transpctrl_icons array.
    enum TranspCtrlIcon : size_t {
        PLAY = 0,
        STOP = 1, /// so far unused
        PAUSE = 2,
        FAST_REWIND = 3,
        FAST_FORWARD = 4,
        ULTRA_FAST_FORWARD = 5,
        ULTRA_FAST_REWIND = 6,
        NONE_COUNT = 7
    };

    struct TranspCtrlIconState {
        TranspCtrlIcon icon;
        float current_anim_time;
        std::chrono::system_clock::time_point start_real_time;
    };

    /**********************************************************************
     * parameters
     **********************************************************************/

    core::param::ParamSlot paramMode;
    core::param::ParamSlot paramAnchor;
    // Custom position
    core::param::ParamSlot paramCustomPosition;
    // Texture Mode
    core::param::ParamSlot paramFileName;
    core::param::ParamSlot paramRelativeWidth;
    // TranspCtrl Icons Mode
    core::param::ParamSlot paramIconColor;
    core::param::ParamSlot paramDuration;
    core::param::ParamSlot paramFastSpeed;
    core::param::ParamSlot paramUltraFastSpeed;
    core::param::ParamSlot paramSpeedParameter;
    core::param::ParamSlot paramTimeParameter;
    // Parameter Mode
    core::param::ParamSlot paramPrefix;
    core::param::ParamSlot paramSuffix;
    core::param::ParamSlot paramParameterName;
    // Label Mode
    core::param::ParamSlot paramText;
    // Font Settings
    core::param::ParamSlot paramFontName;
    core::param::ParamSlot paramFontSize;
    core::param::ParamSlot paramFontColor;

    /**********************************************************************
     * variables
     **********************************************************************/
    GLuint m_texture_id;
    std::unique_ptr<megamol::core::utility::SDFFont> m_font_ptr;
    glm::vec2 m_viewport;
    Rectangle m_current_rectangle;
    // Parameter Mode
    megamol::core::param::AbstractParam* m_parameter_ptr;
    // TranspCtrl Icons
    std::array<GLuint, NONE_COUNT> m_transpctrl_icons;
    TranspCtrlIconState m_state;
    megamol::core::param::AbstractParam* m_speed_parameter_ptr;
    megamol::core::param::AbstractParam* m_time_parameter_ptr;

    /**********************************************************************
     * functions
     **********************************************************************/

    void setParameterGUIVisibility();

    void drawScreenSpaceBillboard(
        glm::mat4 ortho, glm::vec2 viewport, Rectangle rectangle, GLuint texture_id, glm::vec4 overwrite_color);

    void drawScreenSpaceText(glm::mat4 ortho, megamol::core::utility::SDFFont& font, const std::string& text,
        glm::vec4 color, float size, Anchor anchor, Rectangle rectangle) const;

    Rectangle getScreenSpaceRect(glm::vec2 rel_pos, float rel_width, Anchor anchor, unsigned int texture_width,
        unsigned int texture_height, glm::vec2 viewport) const;

    /* parameter callbacks --------------------------------------------- */

    bool onToggleMode(core::param::ParamSlot& slot);
    bool onTextureFileName(core::param::ParamSlot& slot);
    bool onFontName(core::param::ParamSlot& slot);
    bool onParameterName(core::param::ParamSlot& slot);
    bool onTriggerRecalcRectangle(core::param::ParamSlot& slot);
};


} // namespace cinematic_gl
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATIC_OVERLAYRENDERER_H_INCLUDED */
