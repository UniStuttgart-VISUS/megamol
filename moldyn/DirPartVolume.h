/*
 * DirPartVolume.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DIRPARTVOLUME_H_INCLUDED
#define MEGAMOLCORE_DIRPARTVOLUME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Module sampling information derived from directed particles
     */
    class DirPartVolume : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DirPartVolume";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module sampling information derived from directed particles";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        DirPartVolume(void);

        /** Dtor. */
        virtual ~DirPartVolume(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Answer the extend of the data
         *
         * @param caller The calling CallVolumeData
         *
         * @return True on success
         */
        bool outExtend(Call& caller);

        /**
         * Answer the data
         *
         * @param caller The calling CallVolumeData
         *
         * @return True on success
         */
        bool outData(Call& caller);

        /** The call for data */
        CallerSlot inDataSlot;

        /** The call of data */
        CalleeSlot outDataSlot;

        /** Sampling resolution in x direction */
        param::ParamSlot xResSlot;

        /** Sampling resolution in y direction */
        param::ParamSlot yResSlot;

        /** Sampling resolution in z direction */
        param::ParamSlot zResSlot;

        /** Radius of the influence range of each particle in object space */
        param::ParamSlot sampleRadiusSlot;

        /** Button to force a rebuild */
        param::ParamSlot rebuildSlot;

        /** The data hash */
        SIZE_T dataHash;

        /** The frame id */
        unsigned int frameID;

        /** The object space bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The volume data (attribute separated) */
        float *data;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DIRPARTVOLUME_H_INCLUDED */
