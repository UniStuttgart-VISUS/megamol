/*
 * CameraParamsVirtualViewOverride.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAPARAMSVIRTUALVIEWOVERRIDE_H_INCLUDED
#define VISLIB_CAMERAPARAMSVIRTUALVIEWOVERRIDE_H_INCLUDED
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
     * Camera parameter override for the virtual view dimension.
     */
    class CameraParamsVirtualViewOverride : public CameraParamsOverride {

    public:

        /** Ctor. */
        CameraParamsVirtualViewOverride(void);

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base CameraParameters object to use.
         */
        CameraParamsVirtualViewOverride(
            const SmartPtr<CameraParameters>& params);

        /** Dtor. */
        virtual ~CameraParamsVirtualViewOverride(void);

        /**
         * Sets the size of the full virtual view. If the selected clip tile 
         * rectangle covered the whole virtual view the clip tile rectangle 
         * will also be changed that it still covers the whole virtual view. 
         * Otherwise the clip tile rectangle will not be changed.
         *
         * @param viewSize The new size of the full virtual view.
         */
        virtual void SetVirtualViewSize(const ImageSpaceDimension& viewSize);

        /** 
         * Answer the size of the full virtual view.
         *
         * @return The size of the full virtual view.
         */
        virtual const ImageSpaceDimension& VirtualViewSize(void) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsVirtualViewOverride& operator =(
            const CameraParamsVirtualViewOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return true if all members except the syncNumber are equal, or
         *         false if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator ==(const CameraParamsVirtualViewOverride& rhs) const;

    private:

        /**
         * Indicates that a new base object is about to be set.
         *
         * @param params The new base object to be set.
         */
        virtual void preBaseSet(const SmartPtr<CameraParameters>& params);

        /**
         * Resets the override to the base value.
         */
        virtual void resetOverride(void);

        /** Super-class typedef. */
        typedef CameraParamsOverride Super;

        /** The override value for the virtual view. */
        ImageSpaceDimension overrideValue;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMSVIRTUALVIEWOVERRIDE_H_INCLUDED */

