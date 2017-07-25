/*
 * CameraParamOverride.h
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CameraParamOverride_H_INCLUDED
#define MEGAMOLCORE_CameraParamOverride_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/CallRenderView.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsOverride.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Camera parameter override class overriding the eye.
     */
    class MEGAMOLCORE_API CameraParamOverride :
        public vislib::graphics::CameraParamsOverride {

    public:

        /** Ctor. */
        CameraParamOverride(void);

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base 'CameraParameters' object to use.
         */
        CameraParamOverride(
            const vislib::SmartPtr<vislib::graphics::CameraParameters>&
                params);

        /** Dtor. */
        ~CameraParamOverride(void);

        /**
         * Sets the overrides variables based on the call to render the
         * cluster display.
         *
         * @param call The call holding the override values
         */
        void SetOverrides(const view::CallRenderView& call);

        /**
         * Answer the eye for stereo projections.
         *
         * @return The eye for stereo projections.
         */
        virtual vislib::graphics::CameraParameters::StereoEye
        Eye(void) const {
            return this->projOverridden ? this->eye : this->paramsBase()->Eye();
        }

        /** 
         * Answer the type of stereo projection 
         *
         * @return The type of stereo projection 
         */
        virtual vislib::graphics::CameraParameters::ProjectionType
        Projection(void) const {
            return this->projOverridden ? this->pj : this->paramsBase()->Projection();
        }

        /** 
         * Answer the selected clip tile rectangle of the virtual view. (E. g.
         * this should be used as rendering viewport).
         *
         * @return The selected clip tile rectangle 
         */
        virtual const
        vislib::math::Rectangle<vislib::graphics::ImageSpaceType>&
        TileRect(void) const {
            return this->tileOverridden ? this->tile : this->paramsBase()->TileRect();
        }

        /** 
         * Answer the size of the full virtual view.
         *
         * @return The size of the full virtual view.
         */
        virtual const
        vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>&
        VirtualViewSize(void) const {
            return this->tileOverridden ? this->plane : this->paramsBase()->VirtualViewSize();
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamOverride& operator=(
            const CameraParamOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParamOverride& rhs) const;

    private:

        /**
         * Indicates that a new base object is about to be set.
         *
         * @param params The new base object to be set.
         */
        virtual void preBaseSet(
            const vislib::SmartPtr<vislib::graphics::CameraParameters>&
                params);

        /**
         * Resets the override.
         */
        virtual void resetOverride(void);

        /** Flag if the projection is overwritten */
        bool projOverridden;

        /** Flag if the tile is overwritten */
        bool tileOverridden;

        /** eye for stereo projections */
        vislib::graphics::CameraParameters::StereoEye eye;

        /** the projection type */
        vislib::graphics::CameraParameters::ProjectionType pj;

        /** the tile rectangle */
        vislib::math::Rectangle<vislib::graphics::ImageSpaceType> tile;

        /** the viewing plane */
        vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2> plane;

    };

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CameraParamOverride_H_INCLUDED */
