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

#include "mmcore/AbstractGetData3DCall.h"
#include "vislib/math/mathfunctions.h"


namespace megamol {
namespace protein {

/**
 * Get data call for (unclustered) sol-path data
 */
class SolPathDataCall : public megamol::core::AbstractGetData3DCall {
public:
    /**
     * data struct defining the layout of a vertex
     *
     * Linear, interleaved memory layout for fast upload:
     *  x, y, z, speed      (vec4)
     *  time, clusterID     (ivec2)
     */
    typedef struct _vertex_t {

        /** The position of the vertex */
        float x, y, z;

        /** The speed value of the molecule at this vertex */
        float speed;

        /** The frame number of the vertex */
        float time;

        /** The cluster id of the vertex */
        float clusterID;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline struct _vertex_t& operator=(const struct _vertex_t& rhs) {
            this->x = rhs.x;
            this->y = rhs.y;
            this->z = rhs.z;
            this->speed = rhs.speed;
            this->time = rhs.time;
            this->clusterID = rhs.clusterID;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise
         */
        inline bool operator==(const struct _vertex_t& rhs) {
            return vislib::math::IsEqual(this->x, rhs.x) && vislib::math::IsEqual(this->y, rhs.y) &&
                   vislib::math::IsEqual(this->z, rhs.z) && vislib::math::IsEqual(this->speed, rhs.speed) &&
                   (this->time == rhs.time) && (this->clusterID == rhs.clusterID);
        }

    } Vertex;

    /** data structure defining the layout of a pathline */
    typedef struct _pathline_t {

        /** The id of the molecule of this pathline */
        unsigned int id;

        /** The length of the pathline */
        unsigned int length;

        /** The data of the pathline */
        const Vertex* data;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline struct _pathline_t& operator=(const struct _pathline_t& rhs) {
            this->id = rhs.id;
            this->length = rhs.length;
            this->data = rhs.data; // NOT a deep copy
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise
         */
        inline bool operator==(const struct _pathline_t& rhs) {
            return (this->id == rhs.id) && (this->length == rhs.length) && (this->data == rhs.data);
        }

    } Pathline;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "SolPathDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
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
    static const char* FunctionName(unsigned int idx) {
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
    inline unsigned int Count(void) const {
        return this->count;
    }

    /**
     * Answer the maximum speed present
     *
     * @return The maximum speed present
     */
    inline float MaxSpeed(void) const {
        return this->maxSpeed;
    }

    /**
     * Answer the maximum frame number used
     *
     * @return The maximum frame number used
     */
    inline float MaxTime(void) const {
        return this->maxTime;
    }

    /**
     * Answer the minimum speed present
     *
     * @return The minimum speed present
     */
    inline float MinSpeed(void) const {
        return this->minSpeed;
    }

    /**
     * Answer the minimum frame number used
     *
     * @return The minimum frame number used
     */
    inline float MinTime(void) const {
        return this->minTime;
    }

    /**
     * Answer the pathlines
     *
     * @return A pointer to the array of pathlines
     */
    inline const Pathline* Pathlines(void) const {
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
     * @param minTime The minimum frame number used
     * @param maxTime The maximum frame number used
     * @param minSpeed The minimum speed present
     * @param maxSpeed The maximum speed present
     */
    inline void Set(
        unsigned int cnt, const Pathline* lines, float minTime, float maxTime, float minSpeed, float maxSpeed) {
        this->count = cnt;
        this->lines = lines;
        this->minTime = minTime;
        this->maxTime = maxTime;
        this->minSpeed = minSpeed;
        this->maxSpeed = maxSpeed;
    }

private:
    /** The number of pathlines */
    unsigned int count;

    /** The pathlines */
    const Pathline* lines;

    /** The minimum frame number */
    float minTime;

    /** The maximum frame number */
    float maxTime;

    /** The minimum speed */
    float minSpeed;

    /** The maximum speed */
    float maxSpeed;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif /*  MEGAMOL_PROTEIN_SOLPATHDATACALL_H_INCLUDED */
