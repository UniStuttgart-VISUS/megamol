/*
 * CameraParamsTileRectOverride.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAPARAMSTILERECTOVERRIDE_H_INCLUDED
#define VISLIB_CAMERAPARAMSTILERECTOVERRIDE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParamsOverride.h"


namespace vislib {
namespace graphics {


    /**
     * Camera parameter override class overriding the view tile rectangle.
     */
    class CameraParamsTileRectOverride : public CameraParamsOverride {

    public:

        /** Ctor. */
        CameraParamsTileRectOverride(void);

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base 'CameraParameters' object to use.
         */
        CameraParamsTileRectOverride(const SmartPtr<CameraParameters>& params);

        /** Dtor. */
        virtual ~CameraParamsTileRectOverride(void);

        /** resets the clip tile rectangle to the whole virtual view size */
        virtual void ResetTileRect(void);

        /**
         * Sets the selected clip tile rectangle of the virtual view. Also see
         * 'SetVirtualViewSize' for further information.
         *
         * @param tileRect The selected clip tile rectangle of the virtual view
         */
        virtual void SetTileRect(const math::Rectangle<ImageSpaceType>& tileRect);

        /** 
         * Answer the selected clip tile rectangle of the virtual view. (E. g.
         * this should be used as rendering viewport).
         *
         * @return The selected clip tile rectangle 
         */
        virtual const math::Rectangle<ImageSpaceType>& TileRect(void) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsTileRectOverride& operator=(const CameraParamsTileRectOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParamsTileRectOverride& rhs) const;

    private:

        /**
         * Indicates that a new base object is about to be set.
         *
         * @param params The new base object to be set.
         */
        virtual void preBaseSet(const SmartPtr<CameraParameters>& params);

        /**
         * Resets the override.
         */
        virtual void resetOverride(void);

        /** flag indicating if the tile rect covers the whole virtual view */
        bool fullSize;

        /** The selected clip tile rectangle of the virtual view */
        mutable math::Rectangle<ImageSpaceType> tileRect;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMSTILERECTOVERRIDE_H_INCLUDED */

