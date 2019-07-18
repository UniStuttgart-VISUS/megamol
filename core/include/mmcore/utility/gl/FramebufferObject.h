/*
 * FramebufferObject.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_FRAMEBUFFEROBJECT_H_INCLUDED
#define MEGAMOLCORE_FRAMEBUFFEROBJECT_H_INCLUDED

/*	Include space-lion files */
#include "mmcore/utility/gl/Texture2D.h"

/*	Include system libraries */
#include <memory>
#include <string>
#include <vector>

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class FramebufferObject
 *
 * @brief Encapsulates basic framebuffer object functionality.
 *
 * This class encapsulates basic framebuffer object functionality including creation of an FBO,
 * creation and adding of several color attachments and use it for rendering.
 *
 * @author Michael Becher
 */
class FramebufferObject {
public:
    /**
     * Constructor for a FramebufferObject
     */
    FramebufferObject(void);

    /**
     * Destructor
     */
    virtual ~FramebufferObject();

    /* Deleted copy constructor (C++11). Don't wanna go around copying objects with OpenGL handles. */
    FramebufferObject(const FramebufferObject& cpy) = delete;

    /* Deleted move constructor (C++11). Don't wanna go around copying objects with OpenGL handles. */
    FramebufferObject(FramebufferObject&& other) = delete;

    /* Deleted assignment operator (C++11). Don't wanna go around copying objects with OpenGL handles. */
    FramebufferObject& operator=(const FramebufferObject& rhs) = delete;

    /* Deleted move assignment (C++11). Don't wanna go around copying objects with OpenGL handles. */
    FramebufferObject& operator=(FramebufferObject&& rhs) = delete;

    /**
     * @brief Adds one color attachment to the framebuffer.
     * @note New colorbuffers are added at the end of the colorbuffer vector. The index i within the storage vector and
     * the used attachment point GL_COLOR_ATTACHMENTi will be the same.
     * @param internalFormat Specifies channels+size (e.g. GL_R32F)
     * @param format Specifies channels (e.g. GL_RED)
     * @param type Specifies datatype (e.g. GL_FLOAT)
     * @return Returns true if a color attachment was added, false otherwise
     */
    bool createColorAttachment(GLenum internalFormat, GLenum format, GLenum type);

    /**
     * @brief Bind this framebuffer object with all its color attachments
     */
    void bind();

    /**
     * @brief Bind this framebuffer object with a given set of draw buffers
     */
    void bind(const std::vector<GLenum>& draw_buffers);

    /**
     * @brief Bind this framebuffer object with a given set of draw buffers
     */
    void bind(std::vector<GLenum>&& draw_buffers);

    /**
     * @brief Bind the framebuffer to GL_READ_FRAMEBUFFER
     * @param index Set glReadBuffer to color attachment #index or 0, if index > #color attachments
     */
    void bindToRead(unsigned int index);

    /**
     * @brief Bind the framebuffer to GL_DRAW_FRAMEBUFFER using all color attachments for glDrawBuffers
     */
    void bindToDraw();

    /**
     * @brief Bind a color attachment as GL_TEXTURE_2D.
     * @param index Specifies which color is bound. If index > #attachments, the method simply won't bind a texture.
     */
    void bindColorbuffer(unsigned int index);

    /**
     * @brief Bind the depth buffer as GL_TEXTURE_2D.
     * @note Work in progress. Use of this method is discouraged.
     */
    void bindDepthbuffer();

    /**
     * @brief Bind the stencil buffer as GL_TEXTURE_2D.
     * @note Work in progress. Use of this method is higly discouraged.
     */
    void bindStencilbuffer();

    /**
     * @brief Check the framebuffer object for completeness.
     * @return Returns true if the framebuffer object is complete, false otherwise.
     */
    bool checkStatus() const;

    /**
     * Creates a new Framebuffer object with the specified parameters
     *
     * @param width The width of the represented image
     * @param height The height of the represented image
     * @param has_depth True if this FBO should have a depth component, false otherwise
     * @param has_stencil True if this FBO should have a stencil buffer, false otherwise
     * @return True on success, false otherwise
     */
    bool create(int width, int height, bool has_depth = false, bool has_stencil = false);

    /**
     * @brief Resize the framebuffer object, i.e. it's color attachments.
     * @note Might be a bit costly to use often.
     * @param new_width Specifies the new framebuffer width.
     * @param new_width Specifies the new framebuffer height.
     */
    void resize(int new_width, int new_height);

    /**
     * @brief Get the width of the framebuffer object's color attachments
     * @return Returns widths.
     */
    int getWidth() const { return m_width; }

    /**
     * @brief Get the height of the framebuffer object's color attachments
     * @return Returns height.
     */
    int getHeight() const { return m_height; }

    /**
     * @brief Get the log of the framebuffer object
     * @return Returns the log as string
     */
    const std::string& getLog() const { return m_log; }

private:
    /** Handle of the FBO */
    GLuint m_handle;

    /** Colorbuffers attached to the FBO */
    std::vector<std::shared_ptr<Texture2D>> m_colorbuffers;

    /** Handle of the depthbuffer */
    GLuint m_depthbuffer;

    /** Handle of the stencilbuffer */
    GLuint m_stencilbuffer;

    /** Width of the framebuffer i.e. it's color attachments */
    int m_width;

    /** Height of the framebuffer i.e. it's color attachments */
    int m_height;

    /** List of all draw buffer targets (i.e. all color attachments) */
    std::vector<GLenum> m_drawBufs;

    /** The log of the object */
    std::string m_log;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_FRAMEBUFFEROBJECT_H_INCLUDED
