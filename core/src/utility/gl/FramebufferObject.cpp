/*
 * FramebufferObject.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/FramebufferObject.h"

using namespace megamol::core::utility::gl;

/*
 * FramebufferObject::FramebufferObject
 */
FramebufferObject::FramebufferObject(void) : m_width(0), m_height(0) {
    // intentionally empty
}

/*
 * FramebufferObject::~FramebufferObject
 */
FramebufferObject::~FramebufferObject() {
    /*	Delete framebuffer resources. Texture delete themselves when the vector is destroyed. */
    if (m_depthbuffer != 0) glDeleteRenderbuffers(1, &m_depthbuffer);

    /*	Delete framebuffer object */
    if (m_handle != 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
        glDeleteFramebuffers(1, &m_handle);
    }
}

/*
 * FramebufferObject::createColorAttachment
 */
bool FramebufferObject::createColorAttachment(GLenum internalFormat, GLenum format, GLenum type) {
    GLint maxAttachments;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttachments);

    if (m_colorbuffers.size() == (GLuint)maxAttachments) {
        m_log.append("Maximum amount of color attachments reached.\n");
        return false;
    }

    unsigned int bufsSize = static_cast<unsigned int>(m_colorbuffers.size());

    TextureLayout color_attach_layout(internalFormat, m_width, m_height, 1, format, type, 1,
        {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
            {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}},
        {{}});
    std::shared_ptr<Texture2D> new_color_attachment(
        new Texture2D("fbo_" + std::to_string(m_handle) + "_color_attachment_" + std::to_string(bufsSize),
            color_attach_layout, nullptr));
    m_colorbuffers.push_back(new_color_attachment);

    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + bufsSize, GL_TEXTURE_2D, m_colorbuffers.back()->getName(), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_drawBufs.push_back(GL_COLOR_ATTACHMENT0 + bufsSize);

    return true;
}

/*
 * FramebufferObject::bind
 */
void FramebufferObject::bind() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    //	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    //	{
    //		m_log.append("Tried to use incomplete FBO. Fallback to default FBO.\n");
    //		glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //	}
    //	else
    //{
    //	unsigned int bufsSize = static_cast<unsigned int>(m_colorbuffers.size());
    //	GLenum* drawBufs = new GLenum[bufsSize];
    //	for(GLuint i=0; i < bufsSize; i++)
    //	{
    //		drawBufs[i] = (GL_COLOR_ATTACHMENT0+i);
    //	}
    //	glDrawBuffers(bufsSize, drawBufs);
    //
    //	delete drawBufs;
    //}

    glDrawBuffers(static_cast<unsigned int>(m_drawBufs.size()), m_drawBufs.data());
}

/*
 * FramebufferObject::bind
 */
void FramebufferObject::bind(const std::vector<GLenum>& draw_buffers) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);

    glDrawBuffers(static_cast<unsigned int>(draw_buffers.size()), draw_buffers.data());
}

/*
 * FramebufferObject::bind
 */
void FramebufferObject::bind(std::vector<GLenum>&& draw_buffers) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);

    glDrawBuffers(static_cast<unsigned int>(draw_buffers.size()), draw_buffers.data());
}

/*
 * FramebufferObject::bindToRead
 */
void FramebufferObject::bindToRead(unsigned int index) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_handle);
    GLenum readBuffer;
    if (index < static_cast<unsigned int>(m_colorbuffers.size())) readBuffer = (GL_COLOR_ATTACHMENT0 + index);

    glReadBuffer(readBuffer);
}

/*
 * FramebufferObject::bindToDraw
 */
void FramebufferObject::bindToDraw() {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_handle);
    unsigned int bufsSize = static_cast<unsigned int>(m_colorbuffers.size());
    GLenum* drawBufs = new GLenum[bufsSize];
    for (GLuint i = 0; i < bufsSize; i++) {
        drawBufs[i] = (GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(bufsSize, drawBufs);
}

/*
 * FramebufferObject::bindColorbuffer
 */
void FramebufferObject::bindColorbuffer(unsigned int index) {
    if (index < m_colorbuffers.size()) m_colorbuffers[index]->bindTexture();
}

/*
 * FramebufferObject::bindDepthbuffer
 */
void FramebufferObject::bindDepthbuffer() { glBindTexture(GL_TEXTURE_2D, m_depthbuffer); }

/*
 * FramebufferObject::bindStencilbuffer
 */
void FramebufferObject::bindStencilbuffer() { glBindTexture(GL_TEXTURE_2D, m_stencilbuffer); }

/*
 * FramebufferObject::checkStatus
 */
bool FramebufferObject::checkStatus() const {
    if (glCheckFramebufferStatus(m_handle) == GL_FRAMEBUFFER_COMPLETE) return true;
    return false;
}

/*
 * FramebufferObject::create
 */
bool FramebufferObject::create(int width, int height, bool has_depth, bool has_stencil) {
    this->m_width = width;
    this->m_height = height;

    glGenFramebuffers(1, &m_handle);
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);

    if (has_depth) {
        glGenRenderbuffers(1, &m_depthbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, m_depthbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    } else {
        m_depthbuffer = 0;
    }

    // TODO: stencilbuffer
    if (0 && has_stencil) {
        has_stencil = !has_stencil;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

/*
 * FramebufferObject::resize
 */
void FramebufferObject::resize(int new_width, int new_height) {
    m_width = new_width;
    m_height = new_height;

    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    GLenum attachment_point = GL_COLOR_ATTACHMENT0;

    for (auto& colorbuffer : m_colorbuffers) {
        // TODO add more convienient method
        TextureLayout color_attach_layout(colorbuffer->getInternalFormat(), m_width, m_height, 1,
            colorbuffer->getFormat(), colorbuffer->getType(), 1,
            {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
                {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}},
            {});

        colorbuffer->reload(color_attach_layout, nullptr);

        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment_point++, GL_TEXTURE_2D, colorbuffer->getName(), 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // resize depth buffer
    if (m_depthbuffer != 0) {
        glBindRenderbuffer(GL_RENDERBUFFER, m_depthbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, new_width, new_height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
}
