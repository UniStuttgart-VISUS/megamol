/*
 * CameraParamsProjectionOverride.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAPARAMSPROJECTIONOVERRIDE_H_INCLUDED
#define VISLIB_CAMERAPARAMSPROJECTIONOVERRIDE_H_INCLUDED
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
     * Camera parameters for overriding the projection type on specific nodes.
     */
    class CameraParamsProjectionOverride : public CameraParamsOverride {

    public:

        /** 
         * Ctor. 
         *
         * The override will be set to STEREO_PARALLEL as default value.
         */
        CameraParamsProjectionOverride(void);

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base 'CameraParameters' object to use.
         */
        CameraParamsProjectionOverride(const SmartPtr<CameraParameters>& params);

        /** Dtor. */
        ~CameraParamsProjectionOverride(void);

        /** 
         * Answer the type of stereo projection 
         *
         * @return The type of stereo projection 
         */
        virtual ProjectionType Projection(void) const;

        /**
         * Sets the projection type used.
         *
         * @param projectionType The projection type used.
         */
        virtual void SetProjection(ProjectionType projectionType);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsProjectionOverride& operator =(
            const CameraParamsProjectionOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator ==(const CameraParamsProjectionOverride& rhs) const;

    private:

        /** Direct superclass. */
        typedef CameraParamsOverride Super;

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

        /** The override value. */
        ProjectionType projection;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMSPROJECTIONOVERRIDE_H_INCLUDED */

