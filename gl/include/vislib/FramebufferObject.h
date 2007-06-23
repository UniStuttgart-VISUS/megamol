/*
 * FramebufferObject.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FRAMEBUFFEROBJECT_H_INCLUDED
#define VISLIB_FRAMEBUFFEROBJECT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ExtensionsDependent.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * 
     */
    class FramebufferObject : public ExtensionsDependent<FramebufferObject> {

    public:

        /** The possible states for a framebuffer attachment. */
        enum AttachmentState {
            ATTACHMENT_DISABLED = 0,    // Disables the attachment.
            ATTACHMENT_RENDERBUFFER,    // Use a renderbuffer as attachment.
            ATTACHMENT_TEXTURE,         // Use a texture as attachment.
			ATTACHMENT_EXTERNAL_TEXTURE // Use an external texture as 
										// attachment. Use exteme care when 
										// using this, because if the external
										// texture is released before this
										// framebuffer object is released and
										// invalid texture id could be used 
										// resulting in undefined behaviour.
        };

        /** This structure specifies the properties of a color attachment. */
        typedef struct ColorAttachParams_t {
            GLenum internalFormat;      // The internal texture format.
            GLenum format;              // The texture format.
            GLenum type;                // The type of the texture elements.
        } ColorAttachParams;

        /** This structure specifies the properties of a depth attachment. */
        typedef struct DepthAttachParams_t {
            GLenum format;
            AttachmentState state;
			GLuint externalID;			// The openGL ressource ID of the 
										// external Texture, ignored if 
										// state != ATTACHMENT_EXTERNAL_TEXTURE
        } DepthAttachParams;

        /** This structure specifies the properties of a stencil attachment. */
        typedef struct StencilAttachParams_t {
            GLenum format;
            AttachmentState state;
        } StencilAttachParams;

        /**
         * Answer the extensions that are required for framebuffer objects as
         * space-separated ANSI strings.
         *
         * @return The extensions that are requiered for framebuffer objects.
         */
        static const char * RequiredExtensions(void);

        /**
         * Answer the maximum number of color attachment the current hardware
         * supports.
         *
         * @return The number of color attachments the hardware allows.
         *
         * @throws OpenGLException If the number of attachments cannot be 
         *                         retrieved.
         */
        UINT GetMaxColorAttachments(void);

        /** Ctor. */
        FramebufferObject(void);

        /** Dtor. */
        ~FramebufferObject(void);

        /**
         * Bind the texture that is used as render target for the colors.
         * Note that the currently active texture unit is used.
         *
         * The framebuffer object must have been successfully created before 
         * this method can be called.
         *
         * Note: You must set the GL_TEXTURE_MIN_FILTER and 
         * GL_TEXTURE_MAX_FILTER yourself after calling this method. It really
         * only binds the texture.
         *
         * @param which The index of the texture to bind, which must be within
         *              [0, this->GetCntColorAttachments()[.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         *
         * @throws OutOfRangeException   If 'which' does not designate a valid
         *                               color attachment.
         * @throws IllegalStateException If the attachment designated by 'which'
         *                               exists, but is not a texture 
         *                               attachment.
         */
        GLenum BindColorTexture(const UINT which = 0);

        /**
         * Bind the texture that is used as depth target.
         *
         * The framebuffer object must have been successfully created before 
         * this method can be called.
         * 
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         * 
         * @throws IllegalStateException If the framebuffer object has no 
         *                               texture attachment for the depth
         *                               buffer.
         */
        GLenum BindDepthTexture(void);

        /**
         * Create a framebuffer object with the specified dimension having the 
         * specified render targets attached.
         *
         * STENCIL ATTACHMENTS ARE CURRENTLY UNSUPPORTED. sap.state MUST BE
         * ATTACHMENT_DISABLED!
         *
         * @param width               The width of the framebuffer.
         * @param height              The height of the framebuffer.
         * @param cntColorAttachments The number of color attachments to be 
         *                            added. This must be within 
         *                            [0, GetMaxColorAttachments()[.
         * @param cap                 An array of 'cntColorAttachments' color
         *                            attachment specifications.
         * @param dap                 The depth attachment specification.
         * @padam sap                 The stencil attachment specification. THE
         *                            STATE OF 'sap' MUST BE 
         *                            ATTACHMENT_DISABLED!
         *
         * @return true in case of success, false if the framebuffer object 
         *         could be created, but is in incomplete state.
         * 
         * @throws OpenGLException If a resource required for the framebuffer
         *                         object could not be created, e. g. because
         *                         the format of one of the buffers is invalid.
         */
        bool Create(const UINT width, const UINT height, 
            const UINT cntColorAttachments, const ColorAttachParams *cap,
            const DepthAttachParams& dap, const StencilAttachParams& sap);

        /**
         * Create a framebuffer object with the specified dimension having one
         * color attachment and a depth attachment. The depth attachment is
         * realised as renderbuffer object.
         *
         * This is just a convenience method.
         *
         * @param width               The width of the framebuffer.
         * @param height              The height of the framebuffer.
         * @param colorInternalFormat The internal format of the color 
         *                            attachment.
         * @param colorFormat         The format of the color attachment.
         * @param colorType           The datatype of the color attachment.
         * @param depthAttach         The state of the depth attachment.
         * @param depthFormat         The format of the depth attachment.
         * @param stencilAttach       Reserved. Must be ATTACHMENT_DISABLED.
         * @param stencilFormat       Reserved.
         *
         * @return true in case of success, false if the framebuffer object 
         *         could be created, but is in incomplete state.
         * 
         * @throws OpenGLException If a resource required for the framebuffer
         *                         object could not be created, e. g. because
         *                         the format of one of the buffers is invalid.
         */
        inline bool Create(const UINT width, const UINT height, 
                const GLenum colorInternalFormat = GL_RGBA8, 
                const GLenum colorFormat = GL_RGBA, 
                const GLenum colorType = GL_UNSIGNED_BYTE,
                const AttachmentState depthAttach = ATTACHMENT_RENDERBUFFER,
                const GLenum depthFormat = GL_DEPTH_COMPONENT24,
                const AttachmentState stencilAttach = ATTACHMENT_DISABLED,
                const GLenum stencilFormat = GL_STENCIL_INDEX) {
            ColorAttachParams cap;
            cap.internalFormat = colorInternalFormat;
            cap.format = colorFormat;
            cap.type = colorType;
            
            DepthAttachParams dap;
            dap.format = depthFormat;
            dap.state = depthAttach;

            StencilAttachParams sap;
            sap.format = stencilFormat;
            sap.state = stencilAttach;

            return this->Create(width, height, 1, &cap, dap, sap);
        }

		/**
		 * Answer the openGL depth texture ressource ID.
		 *
		 * @return The openGL depth texture ressource ID.
		 *
		 * @throw IllegalStateException if the depth attachment is not a 
		 *        texture attachment.
		 */
		GLuint DepthTextureID(void);

        /**
         * Disable the framebuffer object as render target and restore
         * rendering to the normal window.
         *
         * It is always safe to call this method.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        GLenum Disable(void);

        /**
         * Enable the framebuffer object for rendering to its textures. This
         * method can only be called, if the framebuffer object has been
         * successfully created.
         *
         * The method will also set an appropriate viewport from (0, 0) to
         * (this->GetWidth(), this->GetHeight()). The old viewport will be
         * preserved and automatically restored, when the framebuffer object
         * is disabled. The current draw buffer and read buffer will also be
         * preserved and restored.
         *
         * @param colorAttachment The color attachment to set as render target.
         *                        If no color attachment is attached to the FBO,
         *                        this parameter has no effect.
         *
         * @return GL_NO_ERROR if the FBO has been enabled as render target,
         *         an error code otherwise.
         *
         * @throw OutOfRangeException If an invalid color attachment index was
         *                            specified.
         */
        GLenum Enable(const UINT colorAttachment = 0);

        /**
         * Answer the number of color attachments attached to this framebuffer
         * object.
         *
         * @return The number of color attachments.
         */
        inline UINT GetCntColorAttachments(void) const {
            return this->cntColorAttachments;
        }

        /**
         * Answer the height of the FBO.
         *
         * @return The height.
         */
        inline UINT GetHeight(void) const {
            return static_cast<UINT>(this->height);
        }

        /**
         * Answer the width of the FBO.
         *
         * @return The width.
         */
        inline UINT GetWidth(void) const {
            return static_cast<UINT>(this->width);
        }

        /**
         * Answer whether this object has successfully created a valid
         * framebuffer object.
         *
         * @return true, if the FBO is valid and can be used, false otherwise.
         */
        bool IsValid(void) const throw();

        /**
         * Release all resources allocated for the framebuffer object. The FBO
         * cannot be used any more after this call. It must be recreated before
         * it can be enabled again.
         *
         * @throws OpenGLException If the resources could not be deleted.
         */
        void Release(void);

    private:

        /** This structure defines a framebuffer attachment. */
        typedef struct AttachmentProps_t {
            GLuint id;                  // ID of the OpenGL resource.
            AttachmentState state;      // Type of the resource.
        } AttachmentProps;

        /** Index of the depth attachment in 'attachmentOther'. */
        static const UINT ATTACH_IDX_DEPTH;

        /** Index of the stencil attachment in 'attachmentOther'. */
        static const UINT ATTACH_IDX_STENCIL;

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Always.
         */
        FramebufferObject(const FramebufferObject& rhs);

        /**
         * Create a renderbuffer and allocate its storage using the current
         * dimensions of the FBO.
         *
         * @param outID  Receives the ID of the render buffer created.
         * @param format The format of the render buffer storage to allocate.
         * 
         * @throws OpenGLException If the creation fails.
         */
        void createRenderbuffer(GLuint& outID, const GLenum format);

        /**
         * Creates a texture and sets the appropriate format. The size is
         * determines by the current dimensions of the FBO.
         *
         * @param outID          Receives the ID of the texture.
         * @param internalFormat The internal texture format.
         * @param format         The format of the texture.
         * @param type           The type of the texels.
         *
         * @throws OpenGLException If the creation fails.
         */
        void createTexture(GLuint& outID, const GLenum internalFormat,
            const GLenum format, const GLenum type) const;

        /**
         * Check for completeness of the framebuffer. The framebuffer object 
         * must have been bound before calling this method.
         *
         * @return true, if the framebuffer is complete, false, if the format
         *         is unsupported.
         *
         * @throws OpenGLException If checking the framebuffer status failed,
         *                         e. g. because no valid FBO is bound.
         */
        bool isComplete(void) const;

        /**
         * Save all state changes that are made when enabling the FBO in
         * member variables in order to restore it when disabling the FBO.
         *
         * @throws OpenGLException If the state information cannot be retrieved.
         */
        void saveState(void);

        /**
         * Forbidden assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If this != &rhs.
         */
        FramebufferObject& operator =(const FramebufferObject& rhs);

        /**
         * The properties of the color attachments. This array has 
         * 'cntColorAttachments' elements.
         */
        AttachmentProps *attachmentColor;

        /** The depth and stencil attachment properties (in this order). */
        AttachmentProps attachmentOther[2];

        /** The number of color attachments created for this FBO. */
        UINT cntColorAttachments;

        /** The IDof the frame buffer. */
        GLuint idFb;

        /** The height of the FBO in pixels. */
        GLsizei height;

        /** The draw buffer to restore when disabling an FBO. */
        GLenum oldDrawBuffer;

        /** The read buffer to restore when disabling and FBO. */
        GLenum oldReadBuffer;

        /** The viewport to restore when disabling an FBO. */
        GLint oldVp[4];

        /** The width of the FBO in pixels. */
        GLsizei width;
    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FRAMEBUFFEROBJECT_H_INCLUDED */
