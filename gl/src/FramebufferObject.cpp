/*
 * FramebufferObject.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/FramebufferObject.h"

#include "vislib/glverify.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::graphics::gl::FramebufferObject::RequiredExtensions
 */
const char * vislib::graphics::gl::FramebufferObject::RequiredExtensions(void) {
    return "GL_EXT_framebuffer_object ";
}


/*
 * vislib::graphics::gl::FramebufferObject::GetMaxColorAttachments
 */
UINT vislib::graphics::gl::FramebufferObject::GetMaxColorAttachments(void) {
    USES_GL_VERIFY;
    GLint retval = 0;
    
    GL_VERIFY_THROW(::glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS_EXT, &retval));

    return static_cast<UINT>(retval);
}


/*
 * vislib::graphics::gl::FramebufferObject::FramebufferObject
 */
vislib::graphics::gl::FramebufferObject::FramebufferObject(void) 
        : attachmentColor(NULL), cntColorAttachments(0), idFb(UINT_MAX), 
        height(0), oldDrawBuffer(0), oldReadBuffer(0), width(0) {
    
    this->attachmentOther[0].state = ATTACHMENT_DISABLED;
    this->attachmentOther[1].state = ATTACHMENT_DISABLED;

    ::memset(this->oldVp, 0, sizeof(this->oldVp));
}


/*
 * vislib::graphics::gl::FramebufferObject::~FramebufferObject
 */
vislib::graphics::gl::FramebufferObject::~FramebufferObject(void) {
    try {
        this->Disable();
        this->Release();
    } catch (OpenGLException e) {
        TRACE(Trace::LEVEL_VL_WARN, "\"%s\" at line %d in \"%s\" when "
            "destroying FramebufferObject", e.GetMsgA(), e.GetLine(), 
            e.GetFile());
    }

    // Dtor must ensure deallocation in any case!
    ARY_SAFE_DELETE(this->attachmentColor);
}


/*
 * vislib::graphics::gl::FramebufferObject::BindColorTexture
 */
GLenum vislib::graphics::gl::FramebufferObject::BindColorTexture(
        const UINT which) {
    USES_GL_VERIFY;
    
    if (which < this->cntColorAttachments) {
        if (this->attachmentColor[which].state == ATTACHMENT_TEXTURE) {
            GL_VERIFY_RETURN(::glBindTexture(GL_TEXTURE_2D, 
                this->attachmentColor[which].id));
        } else {
            throw IllegalStateException("The requested color attachment "
                "must be a texture attachment in order to be bound as a "
                "texture.", __FILE__, __LINE__);
        }

    } else {
        throw OutOfRangeException(which, 0, this->cntColorAttachments, 
            __FILE__, __LINE__);
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::FramebufferObject::BindDepthTexture
 */
GLenum vislib::graphics::gl::FramebufferObject::BindDepthTexture(void) {
    USES_GL_VERIFY;

    if ((this->attachmentOther[ATTACH_IDX_DEPTH].state == ATTACHMENT_TEXTURE)
            || (this->attachmentOther[ATTACH_IDX_DEPTH].state 
            == ATTACHMENT_EXTERNAL_TEXTURE)) {
        GL_VERIFY_RETURN(::glBindTexture(GL_TEXTURE_2D, 
            this->attachmentOther[ATTACH_IDX_DEPTH].id));
    } else {
        throw IllegalStateException("The depth attachment must be "
            "a texture attachment in order to be bound as a texture.", 
            __FILE__, __LINE__);
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::FramebufferObject::Create
 */
bool vislib::graphics::gl::FramebufferObject::Create(const UINT width, 
        const UINT height, const UINT cntColorAttachments, 
        const ColorAttachParams *cap, const DepthAttachParams& dap, 
        const StencilAttachParams& sap) {
    USES_GL_DEFERRED_VERIFY;
    bool retval = true;

    /* Release possible old FBO before doing anything else! */
    try {
        this->Release();    // TODO: Could also return false instead of recreate.
    } catch (OpenGLException e) {
        TRACE(Trace::LEVEL_VL_WARN, "Release() of old FBO failed in Create(). "
            "This error is not critical.\n");
    }

    /* Save state changes in attributes. */
    this->width = static_cast<GLsizei>(width);
    this->height = static_cast<GLsizei>(height);
    this->cntColorAttachments = cntColorAttachments;

    /* Initially save state. (In context with Disable TODO). */
    this->saveState();

    /* Create FBO and make it active FBO. */
    GL_DEFERRED_VERIFY(::glGenFramebuffersEXT(1, &this->idFb), __LINE__);
    GL_DEFERRED_VERIFY(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->idFb),
        __LINE__);

    /* Create color buffers and attach it to FBO. */
    this->attachmentColor = new AttachmentProps[this->cntColorAttachments];
    for (UINT i = 0; i < cntColorAttachments; i++) {
        this->attachmentColor[i].state = ATTACHMENT_TEXTURE;
        this->createTexture(this->attachmentColor[i].id, cap[i].internalFormat,
            cap[i].format, cap[i].type);
        GL_DEFERRED_VERIFY(::glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, 
            GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, 
            this->attachmentColor[i].id, 0), __LINE__);
    }

    /* Create the depth buffer. */
    switch (dap.state) {
        case ATTACHMENT_RENDERBUFFER:
            GL_DEFERRED_VERIFY_TRY(this->createRenderbuffer(
                this->attachmentOther[ATTACH_IDX_DEPTH].id, dap.format));
            GL_DEFERRED_VERIFY(::glFramebufferRenderbufferEXT(
                GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
                GL_RENDERBUFFER_EXT,
                this->attachmentOther[ATTACH_IDX_DEPTH].id), __LINE__);
            break;

        case ATTACHMENT_TEXTURE:
            GL_DEFERRED_VERIFY_TRY(this->createTexture(
                this->attachmentOther[ATTACH_IDX_DEPTH].id,
                dap.format, GL_DEPTH_COMPONENT, GL_FLOAT)); 
            // TODO: are other formats than GL_FLOAT supported? Could not find one.
            GL_DEFERRED_VERIFY(::glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, 
                this->attachmentOther[ATTACH_IDX_DEPTH].id, 0), __LINE__);
            break;

        case ATTACHMENT_EXTERNAL_TEXTURE:
            this->attachmentOther[ATTACH_IDX_DEPTH].id = dap.externalID;
            GL_DEFERRED_VERIFY(::glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, 
                this->attachmentOther[ATTACH_IDX_DEPTH].id, 0), __LINE__);
            break;

        default:
            /* Nothing to do. */
            break;
    }
    this->attachmentOther[ATTACH_IDX_DEPTH].state = dap.state;

    /* Create the stencil buffer. */
    //switch (sap.state) {
    //    case ATTACHMENT_RENDERBUFFER:
    //        this->createRenderbuffer(
    //            this->attachmentOther[ATTACH_IDX_STENCIL].id, sap.format);
    //        this->createRenderbuffer(
    //            this->attachmentOther[ATTACH_IDX_STENCIL].id, GL_STENCIL_INDEX);
    //        GL_VERIFY_THROW(::glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT,
    //            GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT,
    //            this->attachmentOther[ATTACH_IDX_STENCIL].id));
    //        break;

    //    case ATTACHMENT_TEXTURE:
    //        this->createTexture(this->attachmentOther[ATTACH_IDX_STENCIL].id,
    //            sap.format, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT);  // TODO: internal format?
    //        //this->createTexture(this->attachmentOther[ATTACH_IDX_STENCIL].id,
    //        //    GL_DEPTH32F_STENCIL8_NV, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT_24_8_NV);
    //        GL_VERIFY_THROW(::glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
    //            GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 
    //            this->attachmentOther[ATTACH_IDX_STENCIL].id, 0));
    //        break;

    //    default:
    //        /* Nothing to do. */
    //        break;
    //}
    //this->attachmentOther[ATTACH_IDX_STENCIL].state = sap.state;
    // TODO: Find format that makes stencil attachment work.
    ASSERT(sap.state == ATTACHMENT_DISABLED);
    this->attachmentOther[ATTACH_IDX_STENCIL].state = ATTACHMENT_DISABLED;

    /* Check for completeness and unbind FBO before returning. */
    retval = this->isComplete();
    GL_DEFERRED_VERIFY(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0), __LINE__);

    /* Check for errors and revert in case one was captured before. */
    if (GL_DEFERRED_FAILED()) {
        GL_DEFERRED_VERIFY_TRY(this->Release());
    }
    
    GL_DEFERRED_VERIFY_THROW(__FILE__);
    return retval;
}


/*
 * vislib::graphics::gl::FramebufferObject::DepthTextureID
 */
GLuint vislib::graphics::gl::FramebufferObject::DepthTextureID(void) {
    if ((this->attachmentOther[ATTACH_IDX_DEPTH].state != ATTACHMENT_TEXTURE)
            && (this->attachmentOther[ATTACH_IDX_DEPTH].state 
            != ATTACHMENT_EXTERNAL_TEXTURE)) {
        throw vislib::IllegalStateException("The depth attachment must be a "
            "texture attachment in order to retrieve the texture id.",
            __FILE__, __LINE__);
    }

    return this->attachmentOther[ATTACH_IDX_DEPTH].id;
}


/*
 * vislib::graphics::gl::FramebufferObject::Disable
 */
GLenum vislib::graphics::gl::FramebufferObject::Disable(void) {
    USES_GL_VERIFY;

    if (::glBindFramebufferEXT == NULL) {
        /* 
         * Extensions might not have been initialised, but dtor will call
         * Disable anyway.
         */
        TRACE(Trace::LEVEL_VL_WARN, "glBindFramebuffer is not available.\n");
        return GL_INVALID_OPERATION;
    }

    GL_VERIFY_RETURN(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

    if (this->IsValid()) {
        // TODO: It would be better to store the enabled state of the FBO.
        GL_VERIFY_RETURN(::glViewport(this->oldVp[0], this->oldVp[1],
            this->oldVp[2], this->oldVp[3]));
        GL_VERIFY_RETURN(::glDrawBuffer(this->oldDrawBuffer));
        GL_VERIFY_RETURN(::glReadBuffer(this->oldReadBuffer));
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::FramebufferObject::Enable
 */
GLenum vislib::graphics::gl::FramebufferObject::Enable(
        const UINT colorAttachment) {
    USES_GL_VERIFY;

    /* Ensure that we enable only valid FBOs. */
    if (!this->IsValid()) {
        TRACE(Trace::LEVEL_VL_ERROR, "Cannot enable invalid FBO.\n");
        return GL_INVALID_OPERATION;
    }

    /* Preserve the state. */
    try {
        this->saveState();
    } catch (OpenGLException e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Could not save OpenGL state before "
            "enabling FBO (\"%s\").\n", e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Bind the FBO and disable interpolation on target textures. */
    GL_VERIFY_RETURN(::glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->idFb));
    GL_VERIFY_RETURN(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
        GL_NEAREST));
    GL_VERIFY_RETURN(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
        GL_NEAREST));

    /* Set the appropriate color attachment as render target. */
    if (this->cntColorAttachments < 1) {
        /* No color attachment, disable draw and read. */
        GL_VERIFY_RETURN(::glDrawBuffer(GL_NONE));
        GL_VERIFY_RETURN(::glReadBuffer(GL_NONE));

    } else if (colorAttachment < this->cntColorAttachments) {
        /* Valid color attachment, set it as draw and read buffer. */
        GL_VERIFY_RETURN(::glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT 
            + colorAttachment));
        GL_VERIFY_RETURN(::glReadBuffer(GL_COLOR_ATTACHMENT0_EXT 
            + colorAttachment));

    } else {
        /* Illegal color attachment id. */
        ASSERT(colorAttachment >= this->cntColorAttachments);
        throw OutOfRangeException(colorAttachment, 0, this->cntColorAttachments,
            __FILE__, __LINE__);
    }

    /* Set viewport. */
    GL_VERIFY_RETURN(::glViewport(0, 0, this->width, this->height));

    return GL_NO_ERROR;
}


/* 
 * vislib::graphics::gl::FramebufferObject::IsValid
 */
bool vislib::graphics::gl::FramebufferObject::IsValid(void) const throw() {
    try {
        return this->isComplete();
    } catch (...) {
        return false;
    }
}


/*
 * vislib::graphics::gl::FramebufferObject::Release
 */
void vislib::graphics::gl::FramebufferObject::Release(void) {
    USES_GL_VERIFY;
    USES_GL_DEFERRED_VERIFY;

    //if (GL_FAILED(this->Disable())) {
    //    TRACE(Trace::LEVEL_VL_WARN, "Disabling FBO before release failed. "
    //        "This is not a critical error.\n");
    //}

    if ((::glDeleteRenderbuffersEXT == NULL) 
            || (::glDeleteFramebuffersEXT == NULL)) {
        /* 
         * Extensions might not have been initialised, but dtor will call 
         * Release anyway. 
         */
        TRACE(Trace::LEVEL_VL_WARN, "glDeleteRenderbuffers or "
            "glDeleteFramebuffers is not available.\n");
        return;
    }

    /* Release depth and stencil buffers, if any. */
    for (UINT i = 0; i < 2; i++) {
        switch (this->attachmentOther[i].state) {

            case ATTACHMENT_TEXTURE:
                if (::glIsTexture(this->attachmentOther[i].id)) {
                    GL_DEFERRED_VERIFY(::glDeleteTextures(1, 
                        &this->attachmentOther[i].id), __LINE__);
                }
                break;

            case ATTACHMENT_RENDERBUFFER:
                if (::glIsRenderbufferEXT(this->attachmentOther[i].id)) {
                    GL_DEFERRED_VERIFY(::glDeleteRenderbuffersEXT(1, 
                        &this->attachmentOther[i].id), __LINE__);
                }
                break;

            case ATTACHMENT_EXTERNAL_TEXTURE:
                /* Nothing to do. */
                break;

            default:
                /* Nothing to do. */
                break;
        }
        this->attachmentOther[i].state = ATTACHMENT_DISABLED;
    }

    /* Release color attachments, if any. */
    for (UINT i = 0; i < this->cntColorAttachments; i++) {
        switch (this->attachmentColor[i].state) {
    
            case ATTACHMENT_TEXTURE:
                if (::glIsTexture(this->attachmentColor[i].id)) {
                    GL_DEFERRED_VERIFY(::glDeleteTextures(1, 
                        &this->attachmentColor[i].id), __LINE__);
                }
                break;

            default:
                /* Should not allocate disabled or RB attachments. */
                ASSERT(false);
                break;
        }
    }
    this->cntColorAttachments = 0;
    ARY_SAFE_DELETE(this->attachmentColor);

    /* Release framebuffer itself. */
    if (::glIsFramebufferEXT(this->idFb)) {
        GL_DEFERRED_VERIFY(::glDeleteFramebuffersEXT(1, &this->idFb), __LINE__);
        this->idFb = UINT_MAX;
    }

    // set width and height to zero to indicate that the fbo is empty
    this->width = 0;
    this->height = 0;

    GL_DEFERRED_VERIFY_THROW(__FILE__);
}


/*
 * vislib::graphics::gl::FramebufferObject::ATTACH_IDX_DEPTH
 */
const UINT vislib::graphics::gl::FramebufferObject::ATTACH_IDX_DEPTH = 0;


/*
 * vislib::graphics::gl::FramebufferObject::ATTACH_IDX_STENCIL
 */
const UINT vislib::graphics::gl::FramebufferObject::ATTACH_IDX_STENCIL = 1;


/*
 * vislib::graphics::gl::FramebufferObject::FramebufferObject
 */
vislib::graphics::gl::FramebufferObject::FramebufferObject(
        const FramebufferObject& rhs) {
    throw UnsupportedOperationException("FramebufferObject::FramebufferObject", 
        __FILE__, __LINE__);
}


/*
 * vislib::graphics::gl::FramebufferObject::createRenderbuffer
 */
void vislib::graphics::gl::FramebufferObject::createRenderbuffer(
        GLuint& outID, const GLenum format) {
    USES_GL_DEFERRED_VERIFY;

    GL_DEFERRED_VERIFY(::glGenRenderbuffersEXT(1, &outID), __LINE__);
    GL_DEFERRED_VERIFY(::glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, outID), 
        __LINE__);
    GL_DEFERRED_VERIFY(::glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, format, 
        this->width, this->height), __LINE__);

    if (GL_DEFERRED_FAILED()) {
        ::glDeleteRenderbuffersEXT(1, &outID);
        outID = UINT_MAX;
    }

    GL_DEFERRED_VERIFY_THROW(__FILE__);
}

/*
 * vislib::graphics::gl::FramebufferObject::createTexture
 */
void vislib::graphics::gl::FramebufferObject::createTexture(GLuint& outID,
        const GLenum internalFormat, const GLenum format, 
        const GLenum type) const {
    USES_GL_DEFERRED_VERIFY;
    GLint oldID;                // Old texture bound for reverting state.
    
    ::glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldID);

    GL_DEFERRED_VERIFY(::glGenTextures(1, &outID), __LINE__);
    GL_DEFERRED_VERIFY(::glBindTexture(GL_TEXTURE_2D, outID), __LINE__);
    GL_DEFERRED_VERIFY(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
        GL_NEAREST), __LINE__);
    GL_DEFERRED_VERIFY(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
        GL_NEAREST), __LINE__);
    GL_DEFERRED_VERIFY(::glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, 
        this->width, this->height, 0, format, type, NULL), __LINE__);

    if (::glIsTexture(oldID)) {
        ::glBindTexture(GL_TEXTURE_2D, oldID);
    }

    if (GL_DEFERRED_FAILED()) {
        ::glDeleteTextures(1, &outID);
        outID = UINT_MAX;
    }

    GL_DEFERRED_VERIFY_THROW(__FILE__);
}


/*
 * vislib::graphics::gl::FramebufferObject::isComplete
 */
bool vislib::graphics::gl::FramebufferObject::isComplete(void) const {
    GLenum status = ::glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    
    switch (status) {

        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
            /* Unreachable. */

        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            TRACE(Trace::LEVEL_ERROR, "The selected framebuffer format is "
                "unsupported.\n");
            /* falls through. */

        default:
            TRACE(Trace::LEVEL_ERROR, "The framebuffer object is not complete "
                "(%u).\n", status);
            return false;
            /* Unreachable. */
    }
}


/* 
 * vislib::graphics::gl::FramebufferObject::saveState
 */
void vislib::graphics::gl::FramebufferObject::saveState(void) {
    USES_GL_VERIFY;
    GLint tmp;

    GL_VERIFY_THROW(::glGetIntegerv(GL_DRAW_BUFFER, &tmp));
    this->oldDrawBuffer = static_cast<GLenum>(tmp);
    
    GL_VERIFY_THROW(::glGetIntegerv(GL_READ_BUFFER, &tmp));
    this->oldReadBuffer = static_cast<GLenum>(tmp);

    GL_VERIFY_THROW(::glGetIntegerv(GL_VIEWPORT, this->oldVp));

    TRACE(Trace::LEVEL_VL_INFO, "FBO saved state:\n"
        "\tGL_DRAW_BUFFER = %d\n"
        "\tGL_READ_BUFFER = %d\n"
        "\tGL_VIEWPORT = %d %d %d %d\n",
        this->oldDrawBuffer, this->oldReadBuffer, this->oldVp[0], 
        this->oldVp[1], this->oldVp[2], this->oldVp[3]);
}


/*
 * vislib::graphics::gl::FramebufferObject::operator =(
 */
vislib::graphics::gl::FramebufferObject& 
vislib::graphics::gl::FramebufferObject::operator =(
        const FramebufferObject& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
