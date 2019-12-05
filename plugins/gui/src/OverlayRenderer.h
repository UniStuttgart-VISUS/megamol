/*
 * OverlayRenderer.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED
#define MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/RendererModule.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"

#include <chrono>
#include <glm/gtc/matrix_transform.hpp>


namespace megamol {
namespace gui {

/**
 * Renders various kinds of overlays.
 */
class OverlayRenderer : public megamol::core::view::RendererModule<megamol::core::view::CallRender3D_2> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "OverlayRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renders various kinds of overlays."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    OverlayRenderer(void);

    /** Dtor. */
    virtual ~OverlayRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core::view::CallRender3D_2& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core::view::CallRender3D_2& call);

private:
    struct TextureData {
        vislib::graphics::gl::OpenGLTexture2D tex;
        unsigned int width;
        unsigned int height;
    };

    struct Rectangle {
        float left;
        float right;
        float top;
        float bottom;
    };

    typedef megamol::core::utility::AbstractFont::Alignment Anchor;

    enum Mode { TEXTURE, TRANSPORT_CTRL, PARAMETER, LABEL };

    // Explicit numbering required as indices in transpctrl_icons array.
    enum TranspCtrlIcon : size_t {
        PLAY = 0,
        STOP = 1, /// unused
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
        float start_anim_time;
        std::chrono::system_clock::time_point start_real_time;
    };

    /**********************************************************************
     * variables
     **********************************************************************/

    TextureData m_texture;
    vislib::graphics::gl::GLSLShader m_shader;
    std::unique_ptr<megamol::core::utility::SDFFont> m_font;
    glm::ivec2 m_viewport;
    Rectangle m_current_rectangle;
    // Parameter Mode
    vislib::SmartPtr<megamol::core::param::AbstractParam> m_parameter_ptr;
    // TranspCtrl Icons
    std::array<TextureData, NONE_COUNT> m_transpctrl_icons;
    TranspCtrlIconState m_state;
    vislib::SmartPtr<megamol::core::param::AbstractParam> m_speed_parameter_ptr;
    vislib::SmartPtr<megamol::core::param::AbstractParam> m_time_parameter_ptr;

    /**********************************************************************
     * functions
     **********************************************************************/

    void setParameterGUIVisibility(void);

    void drawScreenSpaceBillboard(glm::mat4 ortho, Rectangle rectangle, TextureData& texture,
        vislib::graphics::gl::GLSLShader& shader, glm::vec4 overwrite_color) const;

    void drawScreenSpaceText(glm::mat4 ortho, megamol::core::utility::SDFFont& font, const std::string& text,
        glm::vec4 color, float size, Anchor anchor, Rectangle rectangle) const;

    Rectangle getScreenSpaceRect(
        glm::vec2 rel_pos, float rel_width, Anchor anchor, const TextureData& texture, glm::ivec2 viewport) const;

    bool loadTexture(const std::string& filename, TextureData& texture) const;

    bool loadShader(
        vislib::graphics::gl::GLSLShader& shader, const std::string& vert_name, const std::string& frag_name) const;

    size_t loadRawFile(std::string name, void** outData) const;

    /* parameter callbacks --------------------------------------------- */

    bool onToggleMode(core::param::ParamSlot& slot);
    bool onTextureFileName(core::param::ParamSlot& slot);
    bool onFontName(core::param::ParamSlot& slot);
    bool onParameterName(core::param::ParamSlot& slot);
    bool onTriggerRecalcRectangle(core::param::ParamSlot& slot);

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
    core::param::ParamSlot paramSufix;
    core::param::ParamSlot paramParameterName;
    // Label Mode
    core::param::ParamSlot paramText;
    // Font Settings
    core::param::ParamSlot paramFontName;
    core::param::ParamSlot paramFontSize;
    core::param::ParamSlot paramFontColor;
};

} /* end namespace gui */
} /* end namespace megamol */

#endif /* MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED */
