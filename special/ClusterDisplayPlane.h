/*
 * ClusterDisplayPlane.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERDISPLAYPLANE_H_INCLUDED
#define MEGAMOLCORE_CLUSTERDISPLAYPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CoreInstance.h"
#include "vislib/PtrArray.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Class managing a display plane and configuration of all planes
     */
    class ClusterDisplayPlane {
    public:

        /**
         * Possible plane type values
         */
        enum PlaneType {
            TYPE_VOID, // illegal plane
            TYPE_MONO,
            TYPE_STEREO_LEFT,
            TYPE_STEREO_RIGHT,
            TYPE_DOME
        };

        /**
         * Answers the cluster plane object for the specified id. The caller
         * must not delete the returned object!
         *
         * @param id The id of the plane object to be returned
         * @param inst The core instance owning the caller
         *
         * @return The cluster plane object for the given id or NULL if there
         *         is none.
         */
        static const ClusterDisplayPlane *Plane(unsigned int id,
            const CoreInstance& inst);

        /**
         * Ctor.
         * This ctor creates a plane with the given properties and the id
         * zero. The plane will not be cached!
         *
         * @param type The type of the plane
         * @param width The width of the plane
         * @param height The height of the plane
         */
        ClusterDisplayPlane(PlaneType type, float width = 1.0f, float height = 1.0f);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from.
         */
        ClusterDisplayPlane(const ClusterDisplayPlane& src);

        /**
         * Dtor.
         */
        ~ClusterDisplayPlane(void);

        /**
         * Answer the id of the plane.
         *
         * @return The id of the plane
         */
        inline unsigned int Id(void) const {
            return this->id;
        }

        /**
         * Answer the type of the plane.
         *
         * @return The type of the plane
         */
        inline PlaneType Type(void) const {
            return this->planeType;
        }

        /**
         * Answer the width of the virtual viewport of the plane.
         *
         * @return The width of the virtual viewport of the plane
         */
        inline float Width(void) const {
            return this->width;
        }

        /**
         * Answer the height of the virtual viewport of the plane.
         *
         * @return The height of the virtual viewport of the plane
         */
        inline float Height(void) const {
            return this->height;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return Reference to this
         */
        ClusterDisplayPlane& operator=(const ClusterDisplayPlane& rhs);

    private:

        /**
         * The static array storing all planes
         */
        static vislib::PtrArray<ClusterDisplayPlane> planes;

        /**
         * Ctor.
         */
        ClusterDisplayPlane(void);

        /**
         * Loads the configuration for the plane 'id' if available.
         *
         * @param id The id of the plane to load its configuration
         * @param inst The core instance owning the caller
         *
         * @return 'true' on success
         */
        bool loadConfiguration(unsigned int id, const CoreInstance& inst);

        /** The id of this plane */
        unsigned int id;

        /** The type of this plane */
        PlaneType planeType;

        /** The width of the virtual viewport */
        float width;

        /** The height of the virtual viewport */
        float height;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERDISPLAYPLANE_H_INCLUDED */
