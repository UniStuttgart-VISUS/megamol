/*
 * WatermarkRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WATERMARKRENDERER_H_INCLUDED
#define MEGAMOL_GUI_WATERMARKRENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace gui {

using namespace megamol::core;


/**
 * Render watermarks (e.g. logos) in all four corners and the center of the viewport.
 */
class WatermarkRenderer : public view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "WatermarkRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Render watermarks using PNG-files in all four corners of the viewport.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    WatermarkRenderer(void);

    /** Dtor. */
    virtual ~WatermarkRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

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
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core::view::CallRender3D_2& call);

private:
    /**********************************************************************
     * variables
     **********************************************************************/

/** Disable warning for classes not marked for dll export. */
#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */

    enum corner { TOP_LEFT = 0, TOP_RIGHT = 1, BOTTOM_LEFT = 2, BOTTOM_RIGHT = 3, CENTER = 4 };

    vislib::graphics::gl::OpenGLTexture2D textureTopLeft;
    vislib::graphics::gl::OpenGLTexture2D textureTopRight;
    vislib::graphics::gl::OpenGLTexture2D textureBottomLeft;
    vislib::graphics::gl::OpenGLTexture2D textureBottomRight;
    vislib::graphics::gl::OpenGLTexture2D textureCenter;

    glm::vec2 sizeTopLeft;
    glm::vec2 sizeTopRight;
    glm::vec2 sizeBottomLeft;
    glm::vec2 sizeBottomRight;
    glm::vec2 sizeCenter;

    ///////////////////////////////////////////////////////////////

    /** The shader of the font. */
    vislib::graphics::gl::GLSLShader shader;

    /** Vertex buffer object attributes. */
    enum VBOAttribs { POSITION = 0, TEXTURE = 1 };

    /** Vertex buffer object info. */
    struct VBOData {
        GLuint handle;    // buffer handle
        std::string name; // varaible name of attribute in shader
        GLuint index;     // index of attribute location
        unsigned int dim; // dimension of data
    };

    /** Vertex array object. */
    GLuint vaoHandle;

    /** Vertex buffer objects. */
    std::vector<VBOData> vbos;

#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /**********************************************************************
     * functions
     **********************************************************************/

    /**
     * Separate function for loading files from arbitrary paths needed.
     * (not not only from within the resource folder like utility::ResourceWrapper::LoadResource() does)
     */
    SIZE_T loadFile(std::string name, void** outData);

    /** PNG image file must be in RGBA foramt. */
    bool loadTexture(WatermarkRenderer::corner cor, std::string filename);

    /** Draw specified watermark. */
    bool renderWatermark(WatermarkRenderer::corner cor, float vpH, float vpW);

    ///////////////////////////////////////////////////////////////

    /** Load vertex buffer array and vertex buffer objects. */
    bool loadBuffers(void);

    /** Load shaders. */
    bool loadShaders(void);

    /**********************************************************************
     * parameters
     **********************************************************************/

    /** Global. */
    core::param::ParamSlot paramAlpha;
    core::param::ParamSlot paramScaleAll;

    /** Image path parameters. */
    core::param::ParamSlot paramImgTopLeft;
    core::param::ParamSlot paramImgTopRight;
    core::param::ParamSlot paramImgBottomLeft;
    core::param::ParamSlot paramImgBottomRight;
    core::param::ParamSlot paramImgCenter;

    /** Scaling parameters for each watermark. */
    core::param::ParamSlot paramScaleTopLeft;
    core::param::ParamSlot paramScaleTopRight;
    core::param::ParamSlot paramScaleBottomLeft;
    core::param::ParamSlot paramScaleBottomRight;
    core::param::ParamSlot paramScaleCenter;
};

} /* end namespace gui */
} /* end namespace megamol */

#endif /* MEGAMOL_GUI_WATERMARKRENDERER_H_INCLUDED */
