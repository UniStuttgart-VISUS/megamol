/*
 * OverlayRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED
#define MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"

#include <glm/gtc/matrix_transform.hpp>


namespace megamol {
namespace gui {

using namespace megamol::core;


/**
 * Renders various kinds of overlays.
 */
class OverlayRenderer : public view::Renderer3DModule_2 {
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
    typedef megamol::core::utility::AbstractFont::Alignment Anchor;

    enum Mode { TEXTURE, MEDIA_BUTTONS, PARAMETER, LABEL };

    // Explicit numbering for array media_buttons
    enum MediaButton { PLAY = 0, STOP = 1, PAUSE = 2, REWIND = 3, FAST_FORWARD = 4 };

    struct TextureData {
        vislib::graphics::gl::OpenGLTexture2D tex;
        unsigned int width;
        unsigned int height;
    };

    /**********************************************************************
     * variables
     **********************************************************************/

    TextureData texture;
    vislib::graphics::gl::GLSLShader shader;
    std::unique_ptr<megamol::core::utility::SDFFont> font;
    std::array<TextureData, 5> media_buttons;

    /**********************************************************************
     * functions
     **********************************************************************/

    /**
     * Separate function for loading files from arbitrary paths needed.
     * Not only from within the megamol resource folders ...
     */
    size_t loadFile(std::string name, void** outData);

    /** PNG image file must be in RGBA foramt. */
    bool loadTexture(const std::string& fn, TextureData& io_tex);

    /** Load shader. */
    bool loadShader(vislib::graphics::gl::GLSLShader& io_shader);

    void setParameterGUIVisibility(void);

    glm::vec4 getScreenSpaceRect(glm::vec2 rel_pos, float rel_width, Anchor anchor, TextureData& io_tex);

    /* parameter callbacks --------------------------------------------- */

    bool onToggleMode(core::param::ParamSlot& slot);
    bool onTextureFileName(core::param::ParamSlot& slot);
    bool onFontName(core::param::ParamSlot& slot);

    /**********************************************************************
     * parameters
     **********************************************************************/

    core::param::ParamSlot paramMode;
    core::param::ParamSlot paramAnchor;

    core::param::ParamSlot paramCustomPositionSwitch;
    core::param::ParamSlot paramCustomPosition;

    // Texture Mode
    core::param::ParamSlot paramFileName; // callback -> loadTexture

    // Media Buttons Mode
    // core::param::ParamSlot param...

    // Parameter Mode
    core::param::ParamSlot paramPrefix;
    core::param::ParamSlot paramSufix;
    core::param::ParamSlot paramParameterName;

    // Label Mode
    core::param::ParamSlot paramText;
    core::param::ParamSlot paramFont; // enum SDFFont::FontName | callback -> initFont
    core::param::ParamSlot paramFontSize;
};

} /* end namespace gui */
} /* end namespace megamol */

#endif /* MEGAMOL_GUI_OVERLAYRENDERER_H_INCLUDED */
