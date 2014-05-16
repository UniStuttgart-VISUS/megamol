/*
 * ViewDirect3D.h
 *
 * Copyright (C) 2012 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEWDIRECT3D_H_INCLUDED
#define MEGAMOLCORE_VIEWDIRECT3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"

#include "mmd3d.h"
#include "view/View3D.h"

#include "vislib/assert.h"
#include "vislib/CriticalSection.h"
#include "vislib/StackTrace.h"


namespace megamol {
namespace core {
namespace view {

    /**
     * Direct3D 11 view.
     */
    class MEGAMOLCORE_API ViewDirect3D : public View3D {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "ViewDirect3D";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "D3D11 3D View Module";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void);

        /** 
         * Initialises a new instance. 
         */
        ViewDirect3D(void);

        /**
         * Dtor.
         */
        virtual ~ViewDirect3D(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param time The time code of the frame to be displayed
         * @param instTime The instance time code
         */
        virtual void Render(float time, double instTime);

        /**
         * Renders the title scene
         *
         * @param tileX The view tile x coordinate
         * @param tileY The view tile y coordinate
         * @param tileW The view tile width
         * @param tileH The view tile height
         * @param virtW The virtual view width
         * @param virtH The virtual view height
         * @param stereo Flag if stereo rendering is to be performed
         * @param leftEye Flag if the stereo rendering is done for the left eye view
         * @param instTime The instance time code
         * @param core The core
         */
        virtual void Render(float tileX, float tileY, float tileW, float tileH,
            float virtW, float virtH, bool stereo, bool leftEye, double instTime,
            class ::megamol::core::CoreInstance *core);

        /**
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height);

        virtual void UpdateFromContext(mmcRenderViewContext *context);

    protected:

        /**
         * Implementation of 'Module::Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Module::Release'.
         */
        virtual void release(void);

        /** Base class typedef. */
        typedef View3D Base;

    private:

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
        /**
         * Release all D3D resources.
         */
        void finaliseD3D(void);

        /**
         * Prepare all D3D resources on the device.
         */
        void initialiseD3D(ID3D11Device *device);

        /**
         * Handles the switch to a new render target view.
         *
         * This method is called by UpdateFromContext() if the render target 
         * view was changed by the underlying viewer DLL.
         */
        void onResizedD3D(ID3D11RenderTargetView *rtv);

        /**
         * Resizes the D3D resources if the window size changes.
         *
         * This method updates the viewport and releases the old render target
         * view. The render target view is then re-installed in resizedD3D(),
         * which is called once the swap chain was resized.
         */
        void onResizingD3D(const unsigned int width, const unsigned int height);

        /** The device we are using for rendering. */
        ID3D11Device *device;

        /** The depth stencil view (created by ViewDirect3D). */
        ID3D11DepthStencilView *dsv;

        /** The immediate context of 'device'. */
        ID3D11DeviceContext *immediateContext;

        /** The render target view we are rendering to. */
        ID3D11RenderTargetView *rtv;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */

        /** Slot to call the renderer to render */
        CallerSlot updateD3D;
    };

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEWDIRECT3D_H_INCLUDED */
