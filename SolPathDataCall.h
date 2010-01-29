/*
 * SolPathDataCall.h
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_PROTEIN_SOLPATHDATACALL_H_INCLUDED
#define MEGAMOL_PROTEIN_SOLPATHDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGetData3DCall.h"

namespace megamol {
namespace protein {

    /**
     * Get data call for (unclustered) sol-path data
     */
    class SolPathDataCall : public megamol::core::AbstractGetData3DCall {
    public:

        /** data struct defining the layout of a vertex */
        typedef struct _vertex_t {

            /** The position of the vertex */
            float x, y, z;

            /** The speed value of the molecule at this vertex */
            float speed;

            /** The frame number of the vertex */
            unsigned int time;

            /** The cluster id of the vertex */
            unsigned int clusterID;

        } Vertex;

        /** data structure defining the layout of a pathline */
        typedef struct _pathline_t {

            /** The id of the molecule of this pathline */
            unsigned int id;

            /** The length of the pathline */
            unsigned int length;

            /** The data of the pathline */
            Vertex *data;

        } Pathline;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "SolPathDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get solvent path-line data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return megamol::core::AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return megamol::core::AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor */
        SolPathDataCall(void);

        /** Dtor. */
        virtual ~SolPathDataCall(void);

        /**
         * Answer the number of pathlines
         *
         * @return The number of pathlines
         */
        unsigned int Count(void) const {
            return this->count;
        }

        /**
         * Answer the pathlines
         *
         * @return A pointer to the array of pathlines
         */
        const Pathline *Pathlines(void) const {
            return this->lines;
        }

        /**
         * Sets the pathlines data. The object does not take ownership of the
         * array of pathlines. The caller must ensure that the data remains
         * available and the pointer remains valid as long as the data is no
         * longer used.
         *
         * @param cnt The number of pathlines
         * @param lines The array of pathlines
         */
        void Set(unsigned int cnt, Pathline *lines) {
            this->count = cnt;
            this->lines = lines;
        }

    private:

        /** The number of pathlines */
        unsigned int count;

        /** The pathlines */
        Pathline *lines;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /*  MEGAMOL_PROTEIN_SOLPATHDATACALL_H_INCLUDED */
