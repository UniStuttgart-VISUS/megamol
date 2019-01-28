/*
 * RenderOutputOpenGL.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDEROUTPUT_H_INCLUDED
#define MEGAMOLCORE_RENDEROUTPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractRenderOutput.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/Array.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Rectangle.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class MEGAMOLCORE_API RenderOutputOpenGL : public virtual AbstractRenderOutput {
    public:

        /**
         * Deactivates the output buffer
         */
        virtual void DisableOutputBuffer(void);

        /**
         * Activates the output buffer
         */
        virtual void EnableOutputBuffer(void);

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
         * Copies the output buffer settings of 'output'
         *
         * @param call The source object to copy from
         */
        inline void SetOutputBuffer(RenderOutputOpenGL& output) {
            this->outputBuffer = output.outputBuffer;
            this->outputFBO = output.outputFBO;
            this->outputFBOTargets = output.outputFBOTargets;
            this->outputViewport = output.outputViewport;
        }

        /**
         * Sets the opengl built-in output buffer to use. This also sets the
         * framebuffer object to NULL. The viewport to be sued will be set to
         * the current opengl viewport.
         *
         * @param buffer The opengl built-in output buffer to be used
         */
        inline void SetOutputBuffer(GLenum buffer = GL_BACK) {
            GLint vp[4];
            ::glGetIntegerv(GL_VIEWPORT, vp);
            // TODO take the current viewport instead
            this->GetViewport();
            this->SetOutputBuffer(buffer, vp[0], vp[1], vp[2], vp[3]);
        }

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
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo) {
            this->SetOutputBuffer(fbo, 0, NULL, 0, 0,
                fbo->GetWidth(), fbo->GetHeight());
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used. The viewport to be used will be set to the full size
         * of the framebuffer object.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param target The buffer to bind
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT target) {
            this->SetOutputBuffer(fbo, 1, &target, 0, 0,
                fbo->GetWidth(), fbo->GetHeight());
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used. The viewport to be used will be set to the full size
         * of the framebuffer object.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param cntTargets The number of targets 'targets' points to
         * @param targets Pointer to the target identifiers to bind, or NULL if
         *                the default targets should be bound
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT cntTargets,
                UINT* targets) {
            this->SetOutputBuffer(fbo, cntTargets, targets, 0, 0,
                fbo->GetWidth(), fbo->GetHeight());
        }

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
        inline void SetOutputBuffer(GLenum buffer,
                const vislib::math::Rectangle<int>& viewport) {
            this->SetOutputBuffer(buffer, viewport.Left(), viewport.Bottom(),
                viewport.Width(), viewport.Height());
        }

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
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo,
                const vislib::math::Rectangle<int>& viewport) {
            this->SetOutputBuffer(fbo, 0, NULL, viewport.Left(),
                viewport.Bottom(), viewport.Width(), viewport.Height());
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param target The buffer to bind
         * @param viewport The viewport on the output buffer to be used
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT target,
                const vislib::math::Rectangle<int>& viewport) {
            this->SetOutputBuffer(fbo, 1, &target, viewport.Left(),
                viewport.Bottom(), viewport.Width(), viewport.Height());
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param cntTargets The number of targets 'targets' points to
         * @param targets Pointer to the target identifiers to bind, or NULL if
         *                the default targets should be bound
         * @param viewport The viewport on the output buffer to be used
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT cntTargets,
                UINT* targets, const vislib::math::Rectangle<int>& viewport) {
            this->SetOutputBuffer(fbo, cntTargets, targets, viewport.Left(),
                viewport.Bottom(), viewport.Width(), viewport.Height());
        }

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
        inline void SetOutputBuffer(GLenum buffer, int w, int h) {
            this->SetOutputBuffer(buffer, 0, 0, w, h);
        }

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
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, int w, int h) {
            this->SetOutputBuffer(fbo, 0, NULL, 0, 0, w, h);
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param target The buffer to bind
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT target,
                int w, int h) {
            this->SetOutputBuffer(fbo, 1, &target, 0, 0, w, h);
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param cntTargets The number of targets 'targets' points to
         * @param targets Pointer to the target identifiers to bind, or NULL if
         *                the default targets should be bound
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT cntTargets,
                UINT* targets, int w, int h) {
            this->SetOutputBuffer(fbo, cntTargets, targets, 0, 0, w, h);
        }

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
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, int x, int y,
                int w, int h) {
            this->SetOutputBuffer(fbo, 0, NULL, x, y, w, h);
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param target The buffer to bind
         * @param x The x coordinate of the viewport on the output buffer to
         *          be used.
         * @param y The y coordinate of the viewport on the output buffer to
         *          be used.
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        inline void SetOutputBuffer(
                vislib::graphics::gl::FramebufferObject *fbo, UINT target,
                int x, int y, int w, int h) {
            this->SetOutputBuffer(fbo, 1, &target, x, y, w, h);
        }

        /**
         * Sets the output buffer to an framebuffer object. If the framebuffer
         * object is not NULL, it is used, otherwise the opengl built-in output
         * buffer is used.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         * @param cntTargets The number of targets 'targets' points to
         * @param targets Pointer to the target identifiers to bind, or NULL if
         *                the default targets should be bound
         * @param x The x coordinate of the viewport on the output buffer to
         *          be used.
         * @param y The y coordinate of the viewport on the output buffer to
         *          be used.
         * @param w The width of the viewport on the output buffer to be used
         * @param h The height of the viewport on the output buffer to be used
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo,
            UINT cntTargets, UINT* targets, int x, int y, int w, int h);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        RenderOutputOpenGL& operator=(const RenderOutputOpenGL& rhs);

    protected:

        /** Ctor. */
        RenderOutputOpenGL(void);

        /** Dtor. */
        virtual ~RenderOutputOpenGL(void);

    private:

        /** The OpenGL output buffer */
        GLenum outputBuffer;

        /** The framebuffer object the callee should render to */
        vislib::graphics::gl::FramebufferObject *outputFBO;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The ids of the framebuffer object targets to be enabled */
        vislib::Array<UINT> outputFBOTargets;

        /** The viewport on the buffer (mutable for lazy evaluation) */
        mutable vislib::math::Rectangle<int> outputViewport;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDEROUTPUT_H_INCLUDED */
