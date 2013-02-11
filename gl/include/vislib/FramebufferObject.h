/*
 * FramebufferObject.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FRAMEBUFFEROBJECT_H_INCLUDED
#define VISLIB_FRAMEBUFFEROBJECT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ExtensionsDependent.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * This class wraps an OpenGL framebuffer object.
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

        /** This structure specifies the properties of a colour attachment. */
        typedef struct ColourAttachParams_t {
            GLenum internalFormat;      // The internal texture format.
            GLenum format;              // The texture format.
            GLenum type;                // The type of the texture elements.
        } ColourAttachParams, ColorAttachParams;

        /** This structure specifies the properties of a depth attachment. */
        typedef struct DepthAttachParams_t {
            GLenum format;
            AttachmentState state;
            GLuint externalID;          // The openGL ressource ID of the 
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
        static const char *RequiredExtensions(void);

        /**
         * Answer the maximum number of colour attachments the current hardware
         * supports.
         *
         * @return The number of colour attachments the hardware allows.
         *
         * @throws OpenGLException If the number of attachments cannot be 
         *                         retrieved.
         */
        UINT GetMaxColourAttachments(void);

        /** Ctor. */
        FramebufferObject(void);

        /** Dtor. */
        ~FramebufferObject(void);

        /**
         * Bind the texture that is used as render target for the colours.
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
         *              [0, this->GetCntColourAttachments()[.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         *
         * @throws OutOfRangeException   If 'which' does not designate a valid
         *                               colour attachment.
         * @throws IllegalStateException If the attachment designated by 'which'
         *                               exists, but is not a texture 
         *                               attachment.
         */
        GLenum BindColourTexture(const UINT which = 0);

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
         * Once the framebuffer object has been created, the previously bound
         * framebuffer will be restored before the method returns.
         *
         * STENCIL ATTACHMENTS ARE CURRENTLY UNSUPPORTED. sap.state MUST BE
         * ATTACHMENT_DISABLED!
         *
         * @param width                The width of the framebuffer.
         * @param height               The height of the framebuffer.
         * @param cntColourAttachments The number of colour attachments to be 
         *                             added. This must be within 
         *                             [0, GetMaxColourAttachments()[.
         * @param cap                  An array of 'cntColourAttachments' colour
         *                             attachment specifications.
         * @param dap                  The depth attachment specification.
         * @padam sap                  The stencil attachment specification. THE
         *                             STATE OF 'sap' MUST BE 
         *                             ATTACHMENT_DISABLED!
         *
         * @return true in case of success, false if the framebuffer object 
         *         could be created, but is in incomplete state.
         * 
         * @throws OpenGLException If a resource required for the framebuffer
         *                         object could not be created, e. g. because
         *                         the format of one of the buffers is invalid.
         */
        bool Create(const UINT width, const UINT height, 
            const UINT cntColourAttachments, const ColourAttachParams *cap,
            const DepthAttachParams& dap, const StencilAttachParams& sap);

        /**
         * Create a framebuffer object with the specified dimension having one
         * colour attachment and a depth attachment. The depth attachment is
         * realised as renderbuffer object.
         *
         * Once the framebuffer object has been created, the previously bound
         * framebuffer will be restored before the method returns.
         *
         * This is just a convenience method.
         *
         * @param width                The width of the framebuffer.
         * @param height               The height of the framebuffer.
         * @param colourInternalFormat The internal format of the colour 
         *                             attachment.
         * @param colourFormat         The format of the colour attachment.
         * @param colourType           The datatype of the colour attachment.
         * @param depthAttach          The state of the depth attachment.
         * @param depthFormat          The format of the depth attachment.
         * @param stencilAttach        Reserved. Must be ATTACHMENT_DISABLED.
         * @param stencilFormat        Reserved.
         *
         * @return true in case of success, false if the framebuffer object 
         *         could be created, but is in incomplete state.
         * 
         * @throws OpenGLException If a resource required for the framebuffer
         *                         object could not be created, e. g. because
         *                         the format of one of the buffers is invalid.
         */
        inline bool Create(const UINT width, const UINT height, 
                const GLenum colourInternalFormat = GL_RGBA8, 
                const GLenum colourFormat = GL_RGBA, 
                const GLenum colourType = GL_UNSIGNED_BYTE,
                const AttachmentState depthAttach = ATTACHMENT_RENDERBUFFER,
                const GLenum depthFormat = GL_DEPTH_COMPONENT24,
                const AttachmentState stencilAttach = ATTACHMENT_DISABLED,
                const GLenum stencilFormat = GL_STENCIL_INDEX) {
            ColourAttachParams cap;
            cap.internalFormat = colourInternalFormat;
            cap.format = colourFormat;
            cap.type = colourType;
            
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
        inline GLuint DepthTextureID(void) const {
            return this->GetDepthTextureID();
        }

        /**
         * Disable the framebuffer object as render target and restore
         * the previous rendering target and viewport.
         *
         * The FBO will restore its previous state regardless whether it is 
         * actually enabled or not. This helps applications that raise 
         * exceptions in a sequence of nested FBO activations like this:
         *
         * fbo1->Enable();
         * try {
         *     fbo2->Enable();
         *     throw Exception(__FILE__, __LINE__);
         *     fbo2->Disable();
         * } catch(...) { }
         * fbo1->Disable();
         *
         * The last call will restore the state before 'fbo1' was enabled
         * although 'fbo1' is not enabled at this point ('fbo2' is).
         *
         * It is always safe to call this method.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        GLenum Disable(void) throw();

        /**
         * Draw the 'colourAttachment'th colour attachment as screen-filling
         * rectangle. The 'colourAttachment'th colour attachment must be a 
         * texture attachment for this method to succeed.
         *
         * This method preserved the 2D texture state, the 2D texture binding 
         * state and matrix stack contents.
         *
         * @param colourAttachment The colour attachment to be drawn. Defaults 
         *                         to 0.
         * @param minFilter        The texture filtering used for minification.
         *                         Defaults to GL_LINEAR.
         * @param magFilter        The texture filtering used for magnification.
         *                         Defaults to GL_LINEAR.
         * @param depth            The depth in z-direction of the rectangle 
         *                         drawn.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         *
         * @throw IllegalStateException If no colour texture is attached.
         * @throw OutOfRangeException If colour textures are attached, but the
         *                            index 'colourAttachment' does not 
         *                            designate a legal attachment.
         */
        inline GLenum DrawColourTexture(const UINT colourAttachment = 0,
                const GLint minFilter = GL_LINEAR, 
                const GLint magFilter = GL_LINEAR,
                const double depth = 0.5) const {
            return this->drawTexture(this->GetColourTextureID(
                colourAttachment), minFilter, magFilter, depth);
        }

        /**
         * Draw the depth attachment as screen-filling rectangle. The depth 
         * attachment must be a texture attachment for this method to succeed.
         *
         * This method preserved the 2D texture state, the 2D texture binding 
         * state and matrix stack contents.
         *
         * @param minFilter The texture filtering used for minification.
         *                  Defaults to GL_LINEAR.
         * @param magFilter The texture filtering used for magnification.
         *                  Defaults to GL_LINEAR.
         * @param depth     The depth in z-direction of the rectangle drawn.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         *
         * @throw IllegalStateException If no depth texture attachment exists.
         */
        inline GLenum DrawDepthTexture(const GLint minFilter = GL_LINEAR,
                const GLint magFilter = GL_LINEAR, 
                const double depth = 0.5) const {
            return this->drawTexture(this->GetDepthTextureID(), minFilter, 
                magFilter, depth);
        }

        /**
         * Enable the framebuffer object for rendering to its textures. This
         * method can only be called, if the framebuffer object has been
         * successfully created.
         *
         * The method has no effect if the FBO is already the current render
         * target, i.e. if IsEnabled() returns true. In this case, the method
         * returns GL_NO_ERROR immediately.
         *
         * The method will also set an appropriate viewport from (0, 0) to
         * (this->GetWidth(), this->GetHeight()). The old viewport will be
         * preserved and automatically restored, when the framebuffer object
         * is disabled. The current draw buffer and read buffer will also be
         * preserved and restored.
         *
         * @param colourAttachment The colour attachment to set as render 
         *                         target. If no color attachment is attached 
         *                         to the FBO, this parameter has no effect.
         *
         * @return GL_NO_ERROR if the FBO has been enabled as render target,
         *         an error code otherwise.
         *
         * @throw OutOfRangeException If an invalid colour attachment index was
         *                            specified.
         */
        GLenum Enable(const UINT colourAttachment = 0);

        /**
         * TODO: Document
         * For multiple render targets.
         * Does not set and unsets read buffer!
         * OpenGL morons defined GL_COLOR_ATTACHMENT0_EXT to use
         */
        GLenum EnableMultipleV(UINT cntColourAttachments,
            const UINT* colourAttachments);

        /**
         * TODO: Document
         * For multiple render targets
         * Does not set and unsets read buffer!
         * OpenGL morons defined GL_COLOR_ATTACHMENT0_EXT to use
         */
        GLenum EnableMultiple(UINT cntColourAttachments, ...);

        /**
         * Answer the number of colour attachments attached to this framebuffer
         * object.
         *
         * @return The number of colour attachments.
         */
        inline UINT GetCntColourAttachments(void) const {
            return this->cntColourAttachments;
        }

        /**
         * Read the pixel data from the 'colourAttachment''th colour attachment.
         *
         * @param outData          Receives the texture data. The caller is 
         *                         responsible for ensuring that the buffer is
         *                         large enough. The caller remains owner of the
         *                         buffer.
         * @param colourAttachment The colour attachment to retrieve the ID of.
         * @param format           A pixel format for the returned data. 
         * @param type             A pixel type for the returned data. 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        inline GLenum GetColourTexture(void *outData, 
                const UINT colourAttachment = 0, const GLenum format = GL_RGBA, 
                const GLenum type = GL_UNSIGNED_BYTE) {
            return this->readTexture(outData, this->GetColourTextureID(
                colourAttachment), format, type);
        }

        /**
         * Answer the OpenGL resource ID of the 'colourAttachment'th colour
         * attachment if one is attached.
         *
         * NOTE: IT IS UNSAFE TO MANIPULATE THE TEXTURE USING THE RETURNED ID!
         *
         * @param colourAttachment The colour attachment to retrieve the ID of.
         *
         * @return The OpenGL resource ID of the colour texture.
         *
         * @throw IllegalStateException If no colour texture is attached.
         * @throw OutOfRangeException If colour textures are attached, but the
         *                            index 'colourAttachment' does not 
         *                            designate a legal attachment.
         */
        GLuint GetColourTextureID(const UINT colourAttachment = 0) const;

        /**
         * Read the pixel data from the depth attachment.
         *
         * @param outData          Receives the texture data. The caller is 
         *                         responsible for ensuring that the buffer is
         *                         large enough. The caller remains owner of the
         *                         buffer.
         * @param format           A pixel format for the returned data. 
         * @param type             A pixel type for the returned data. 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        inline GLenum GetDepthTexture(void *outData, 
                const GLenum format = GL_RGBA, 
                const GLenum type = GL_UNSIGNED_BYTE) {
            return this->readTexture(outData, this->GetDepthTextureID(),
                format, type);
        }

        /**
         * Answer the OpenGL resource ID of the depth texture if one is 
         * attached.
         *
         * NOTE: IT IS UNSAFE TO MANIPULATE THE TEXTURE USING THE RETURNED ID!
         *
         * @return The OpenGL resource ID of the depth texture.
         *
         * @throw IllegalStateException If no depth texture is attached.
         */
        GLuint GetDepthTextureID(void) const;

        /**
         * Answer the height of the FBO.
         *
         * @return The height.
         */
        inline UINT GetHeight(void) const {
            return static_cast<UINT>(this->height);
        }

        /**
         * Answer the OpenGL ID of the frame buffer object. This ID is only 
         * valid if the frame buffer object has been successfully created 
         * before.
         *
         * NOTE: IT IS UNSAFE TO MANIPULATE THE FRAME BUFFER OBJECT USING THE
         * RETURNED ID!
         *
         * @return The OpenGL resource ID of the frame buffer object.
         */
        inline GLuint GetID(void) const {
            return this->idFb;
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
         * Answer whether the FBO is the currently active render target.
         *
         * @return true if the FBO is the currently active render target, 
         *         false otherwise.
         *
         * @throws OpenGLException If the information could not be retrieved
         *                         from the state machine.
         */
        bool IsEnabled(void);

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
         * If the creation fails, the method will delete any created resources
         * by itself before returning.
         *
         * @param outID  Receives the ID of the render buffer created.
         * @param format The format of the render buffer storage to allocate.
         * 
         * @throws OpenGLException In case of an error
         */
        void createRenderbuffer(GLuint& outID, const GLenum format);

        /**
         * Creates a texture and sets the appropriate format. The size is
         * determines by the current dimensions of the FBO.
         *
         * If the creation fails, the method will delete any created resources
         * by itself before returning.
         *
         * @param outID          Receives the ID of the texture.
         * @param internalFormat The internal texture format.
         * @param format         The format of the texture.
         * @param type           The type of the texels.
         *
         * @throws OpenGLException In case of an error
         */
        void createTexture(GLuint& outID, const GLenum internalFormat,
            const GLenum format, const GLenum type) const;

        /**
         * Draw a fullscreen rectangle showing the 2D texture with resource ID 
         * 'id'.
         *
         * This method preserved the 2D texture state, the 2D texture binding 
         * state and matrix stack contents.
         *
         * @param id        The ID of a 2D texture.
         * @param minFilter The texture filtering used for minification.
         * @param magFilter The texture filtering used for magnification.
         * @param depth     The depth in z-direction of the rectangle drawn.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        GLenum drawTexture(const GLuint id, const GLint minFilter, 
            const GLint magFilter, const double depth) const;

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
         * Read the pixel data as pixels with format 'format' and elements of
         * type 'type' into 'outData'.
         *
         * @param outData Receives the texture data.
         * @param id      The ID of the texture to retrieve.
         * @param format  A pixel format for the returned data. 
         * @param type    A pixel type for the returned data. 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        GLenum readTexture(void *outData, const GLuint id, const GLenum format,
            const GLenum type);

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
         * The properties of the colour attachments. This array has 
         * 'cntColourAttachments' elements.
         */
        AttachmentProps *attachmentColour;

        /** The depth and stencil attachment properties (in this order). */
        AttachmentProps attachmentOther[2];

        /** The number of colour attachments created for this FBO. */
        UINT cntColourAttachments;

        /** The ID of the frame buffer. */
        GLuint idFb;

        /** The height of the FBO in pixels. */
        GLsizei height;

        /** The draw buffer to restore when disabling an FBO. */
        GLenum oldDrawBuffer;

        /**
         * Remembers the ID of the FB that was active before the FBO was 
         * enabled. 
         */
        GLint oldFb;

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
