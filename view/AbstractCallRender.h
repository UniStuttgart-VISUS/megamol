/*
 * AbstractCallRender.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "vislib/FramebufferObject.h"
#include "vislib/Rectangle.h"
#include <GL/gl.h>


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class MEGAMOLCORE_API AbstractCallRender : public Call {
    public:

        /** Dtor. */
        virtual ~AbstractCallRender(void);

        /**
         * Deactivates the output buffer
         */
        void DisableOutputBuffer(void);

        /**
         * Activates the output buffer
         */
        void EnableOutputBuffer(void);

        /**
         * Answer the framebuffer object to be used.
         *
         * @return The framebuffer object to be used
         */
        vislib::graphics::gl::FramebufferObject *FrameBufferObject(void) const;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /**
         * Answer the viewport on the output buffer to be used.
         *
         * @return The viewport on the output buffer to be used
         */
        const vislib::math::Rectangle<int>& GetViewport(void) const;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /**
         * Answer the opengl built-in output buffer to be used.
         *
         * @return The opengl built-in output buffer to be used
         */
        GLenum OutputBuffer(void) const;

        /**
         * Resets the output buffer setting to use the opengl built-in output
         * buffer 'BACK'.
         */
        void ResetOutputBuffer(void);

        /**
         * Sets the opengl built-in output buffer to use. This also sets the
         * framebuffer object to NULL. The viewport to be sued will be set to
         * the current opengl viewport.
         *
         * @param buffer The opengl built-in output buffer to be used
         */
        void SetOutputBuffer(GLenum buffer = GL_BACK);

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used. The viewport to be used will be set to the full size
         * of the framebuffer object.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo);

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /**
         * Sets the opengl built-in output buffer to use. This also sets the
         * framebuffer object to NULL.
         *
         * @param buffer The opengl built-in output buffer to be used
         * @param viewport The viewport on the output buffer to be used
         */
        void SetOutputBuffer(GLenum buffer,
            const vislib::math::Rectangle<int>& viewport);

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param viewport The viewport on the output buffer to be used
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo,
            const vislib::math::Rectangle<int>& viewport);
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /**
         * Sets the opengl built-in output buffer to use. This also sets the
         * framebuffer object to NULL.
         *
         * @param buffer The opengl built-in output buffer to be used
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        void SetOutputBuffer(GLenum buffer, int w, int h);

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo,
            int w, int h);

        /**
         * Sets the opengl built-in output buffer to use. This also sets the
         * framebuffer object to NULL.
         *
         * @param buffer The opengl built-in output buffer to be used
         * @param x The x coordinate of the viewport on the output buffer to
         *          be used.
         * @param y The y coordinate of the viewport on the output buffer to
         *          be used.
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        void SetOutputBuffer(GLenum buffer, int x, int y, int w, int h);

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param x The x coordinate of the viewport on the output buffer to
         *          be used.
         * @param y The y coordinate of the viewport on the output buffer to
         *          be used.
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo,
            int x, int y, int w, int h);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        AbstractCallRender& operator=(const AbstractCallRender& rhs);

    protected:

        /** Ctor. */
        AbstractCallRender(void);

    private:

        /** The OpenGL output buffer */
        GLenum outputBuffer;

        /** The framebuffer object the callee should render to */
        vislib::graphics::gl::FramebufferObject *outputFBO;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The viewport on the buffer (mutable for lazy evaluation) */
        mutable vislib::math::Rectangle<int> outputViewport;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED */
