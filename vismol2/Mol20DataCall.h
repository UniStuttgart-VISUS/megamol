/*
 * Mol20DataCall.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MOL20DATACALL_H_INCLUDED
#define MEGAMOLCORE_MOL20DATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "vismol2/Mol20DataSource.h"
#include "vislib/Cuboid.h"


namespace megamol {
namespace core {
namespace vismol2 {


    /**
     * Call for mol 2.0 data
     */
	class Mol20DataCall : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "Mol20DataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get mol 2.0 data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "GetData";
                case 1: return "GetExtent";
            }
            return NULL;
        }

        /** Ctor. */
        Mol20DataCall(void);

        /** Dtor. */
        virtual ~Mol20DataCall(void);

        /**
         * Answers the bounding box of the stored data.
         *
         * @return The bounding box of the stored data.
         */
        inline const vislib::math::Cuboid<float>& BoundingBox(void) const {
            return this->bbox;
        }

        /**
         * Answers the stored frame data
         *
         * @return The stored frame data
         */
        inline Mol20DataSource::Frame *Frame(void) const {
            return this->frame;
        }

        /**
         * Sets the bounding box of the stored data.
         *
         * @param bbox The new bounding box of the stored data.
         */
        inline void SetBoundingBox(const vislib::math::Cuboid<float>& bbox) {
            this->bbox = bbox;
        }

        /**
         * Sets the frame data.
         *
         * @param frame The new frame data.
         * @param unlockOld If 'true' calls 'Unlock' on the previously stored
         *                  frame data.
         */
        inline void SetFrameData(Mol20DataSource::Frame *frame,
                bool unlockOld = true) {
            if ((this->frame != NULL) && unlockOld) {
                this->frame->Unlock();
            }
            this->frame = frame;
        }

        /**
         * Sets the time.
         *
         * @param t The new time.
         */
        inline void SetTime(unsigned int t) {
            this->time = t;
        }

        /**
         * Answers the time of the stored data.
         *
         * @return The time of the stored data.
         */
        inline unsigned int Time(void) const {
            return this->time;
        }

    private:

        /** the stored bounding box */
        vislib::math::Cuboid<float> bbox;

        /** the stored frame */
        Mol20DataSource::Frame *frame;

        /** the stored time */
        unsigned int time;

    };

    /** Description class typedef */
    typedef CallAutoDescription<Mol20DataCall> Mol20DataCallDescription;


} /* end namespace vismol2 */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOL20DATACALL_H_INCLUDED */
