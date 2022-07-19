/*
 * misc/BezierCurvesListDataCall.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/assert.h"


namespace megamol::geocalls {

/**
 * Call for lists cubic bézier curve data.
 * Each curve is specified by a data array of control points and an index
 * array of specifying the quadtruples of each curve.
 * Each curve is defined by four control points p0, p1, p2, and p3, where
 * p0 and p3 are interpolated control points at the ends of the curve and
 * p1 and p2 are approximated control points.
 * The vector (p1 - p0) forms the tangent at p0.
 * The vector (p3 - p2) forms the tangent at p3.
 */
class BezierCurvesListDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Supported data layouts for the point data
     */
    typedef enum _DataLayout_t {
        DATALAYOUT_NONE,
        DATALAYOUT_XYZ_F,
        DATALAYOUT_XYZ_F_RGB_B,
        DATALAYOUT_XYZR_F_RGB_B,
        DATALAYOUT_XYZR_F
    } DataLayout;

    /**
     * Nested class of one list of curves
     */
    class Curves {
    public:
        /**
         * Ctor
         */
        Curves(void);

        /**
         * Copy Ctor. Does take ownership of the memory!
         *
         * @param src The object to clone from.
         */
        Curves(const Curves& src);

        /**
         * Dtor
         */
        ~Curves(void);

        /**
         * Clears all data
         */
        void Clear(void);

        /**
         * Gets the data layout
         *
         * @return The data layout
         */
        inline DataLayout GetDataLayout(void) const {
            return this->layout;
        }

        /**
         * Gets the data pointer
         *
         * @param T the type to return the data pointer in
         *
         * @return The data pointer
         */
        template<class T>
        inline const T* GetData(void) const {
            return reinterpret_cast<const T*>(this->data);
        }

        /**
         * Gets the data pointer at the specific location
         *
         * @param T the type to return the data pointer in
         * @param offset The offset in bytes to the data pointer
         *
         * @return The requested data pointer
         */
        template<class T>
        inline const T* GetDataAt(size_t offset) const {
            return reinterpret_cast<const T*>(this->data + offset);
        }

        /**
         * Gets the data pointer
         *
         * @return The data pointer
         */
        inline const void* GetData(void) const {
            return static_cast<const void*>(this->data);
        }

        /**
         * Gets the data pointer at the specific location
         *
         * @param offset The offset in bytes to the data pointer
         *
         * @return The requested data pointer
         */
        inline const void* GetDataAt(size_t offset) const {
            return static_cast<const void*>(this->data + offset);
        }

        /**
         * Gets the number of points (not bytes!) stored in the memory of
         * 'data'.
         *
         * @return The number of points in 'data'
         */
        inline size_t GetDataPointCount(void) const {
            return this->data_cnt;
        }

        /**
         * Gets the index pointer
         *
         * @return The index pointer
         */
        inline const unsigned int* GetIndex(void) const {
            return this->index;
        }

        /**
         * Gets the number of indices stored in 'index'. The value will be
         * a multiple of four.
         *
         * @return The number of indices stored in 'index'
         */
        inline size_t GetIndexCount(void) const {
            return this->index_cnt;
        }

        /**
         * Gets the global radius
         *
         * @return The global radius
         */
        inline float GetGlobalRadius(void) const {
            return this->rad;
        }

        /**
         * Gets the global colour as pointer to RGB
         *
         * @return The global colour
         */
        inline const unsigned char* GetGlobalColour(void) const {
            return this->col;
        }

        /**
         * Sets the global radius
         *
         * @param r The new global radius
         */
        inline void SetGlobalRadius(float r) {
            this->rad = r;
        }

        /**
         * Sets the global colour
         *
         * @param r Red [0..255]
         * @param g Green [0..255]
         * @param b Blue [0..255]
         */
        inline void SetGlobalColour(unsigned char r, unsigned char g, unsigned char b) {
            this->col[0] = r;
            this->col[1] = g;
            this->col[2] = b;
        }

        /**
         * Sets the data of this object
         *
         * @param layout The data layout
         * @param data The data pointer.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be NULL.
         * @param data_cnt The number of points (not bytes!) stored in
         *            'data'.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be zero.
         * @param data_memory_ownership If set to true, this object will
         *            take ownership of the memory 'data' points to.
         *            Especially the memory will be freed when this object
         *            is destroyed.
         *            Using 'delete[] static_cast<unsigned char*>(data);'
         *            If set to false, the memory ownership is not taken,
         *            i.e. the caller must ensure the pointers remain
         *            valid as long as they are used by this object.
         * @param index The index pointer.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be NULL.
         * @param index_cnt The number of entries (not bytes) in the index
         *            array. Must be a multiple of four.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be zero.
         * @param index_memory_ownership If set to true, this object will
         *            take ownership of the memory 'index' points to.
         *            Especially the memory will be freed when this object
         *            is destroyed. Using 'delete[] index;'
         *            If set to false, the memory ownership is not taken,
         *            i.e. the caller must ensure the pointers remain
         *            valid as long as they are used by this object.
         * @param rad The global radius to be used
         * @param colR The red component of the global colour to be used
         * @param colG The green component of the global colour to be used
         * @param colB The blue component of the global colour to be used
         */
        inline void Set(DataLayout layout, const void* data, size_t data_cnt, bool data_memory_ownership,
            const unsigned int* index, size_t index_cnt, bool index_memory_ownership, float rad, unsigned char colR,
            unsigned char colG, unsigned char colB) {
            this->Clear();
            this->layout = layout;
            this->data = static_cast<const unsigned char*>(data);
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->data != NULL));
            this->data_cnt = data_cnt;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->data_cnt > 0));
            this->data_memory_ownership = data_memory_ownership;
            this->index = index;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->index != NULL));
            this->index_cnt = index_cnt;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->index_cnt > 0));
            ASSERT((this->index_cnt % 4) == 0);
            this->index_memory_ownership = index_memory_ownership;
            this->rad = rad;
            this->col[0] = colR;
            this->col[1] = colG;
            this->col[2] = colB;
        }

        /**
         * Sets the data of this object, not taking ownership of the
         * provided memories and not specifying global values.
         *
         * @param layout The data layout
         * @param data The data pointer.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be NULL.
         * @param data_cnt The number of points (not bytes!) stored in
         *            'data'.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be zero.
         * @param index The index pointer.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be NULL.
         * @param index_cnt The number of entries (not bytes) in the index
         *            array. Must be a multiple of four.
         *            If the layout is not DATALAYOUT_NONE, this must not
         *            be zero.
         */
        inline void Set(
            DataLayout layout, const void* data, size_t data_cnt, const unsigned int* index, size_t index_cnt) {
            this->Clear();
            this->layout = layout;
            this->data = static_cast<const unsigned char*>(data);
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->data != NULL));
            this->data_cnt = data_cnt;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->data_cnt > 0));
            this->data_memory_ownership = false;
            this->index = index;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->index != NULL));
            this->index_cnt = index_cnt;
            ASSERT((this->layout == DATALAYOUT_NONE) || (this->index_cnt > 0));
            ASSERT((this->index_cnt % 4) == 0);
            this->index_memory_ownership = false;
        }

        /**
         * Assignment. Does not take memory ownership!
         *
         * @param rhs The right hand side operand
         */
        inline void Set(const Curves& rhs) {
            *this = rhs;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand
         * @param take_memory If set to 'true' and 'rhs' owns the memory
         *                    of it's data, the memory ownership is
         *                    transferred to this object
         */
        inline void Set(Curves& rhs, bool take_memory) {
            *this = rhs;
            if (take_memory) {
                if (rhs.data_memory_ownership) {
                    rhs.data_memory_ownership = false;
                    this->data_memory_ownership = true;
                }
                if (rhs.index_memory_ownership) {
                    rhs.index_memory_ownership = false;
                    this->index_memory_ownership = true;
                }
            }
        }

        /**
         * Test for equality. Only compares the pointers.
         *
         * @param rhs The right hand side operand
         *
         * @return True if the members of 'rhs' and 'this' are equal
         */
        inline bool operator==(const Curves& rhs) const {
            return (this->layout == rhs.layout) && (this->data == rhs.data) &&
                   (this->data_memory_ownership == rhs.data_memory_ownership) && (this->data_cnt == rhs.data_cnt) &&
                   (this->index == rhs.index) && (this->index_memory_ownership == rhs.index_memory_ownership) &&
                   (this->index_cnt == rhs.index_cnt) && (this->rad == rhs.rad) && (this->col[0] == rhs.col[0]) &&
                   (this->col[1] == rhs.col[1]) && (this->col[2] == rhs.col[2]);
        }

        /**
         * Assignment operator. Does not take memory ownership!
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline Curves& operator=(const Curves& rhs) {
            if (this == &rhs)
                return *this;
            this->Clear();
            this->layout = rhs.layout;
            this->data = rhs.data;
            this->data_memory_ownership = false;
            this->data_cnt = rhs.data_cnt;
            this->index = rhs.index;
            this->index_memory_ownership = false;
            this->index_cnt = rhs.index_cnt;
            this->rad = rhs.rad;
            this->col[0] = rhs.col[0];
            this->col[1] = rhs.col[1];
            this->col[2] = rhs.col[2];
            return *this;
        }

    private:
        /** The data layout */
        DataLayout layout;

        /** The data array */
        const unsigned char* data;

        /**
         * Flag whether or not the memory of 'data' is owned by this
         * object
         */
        bool data_memory_ownership;

        /** Number of points stored in 'data' */
        size_t data_cnt;

        /** The index array */
        const unsigned int* index;

        /**
         * Flag whether or not the memory of 'index' is owned by this
         * object
         */
        bool index_memory_ownership;

        /** Number of indices (ints) stored in 'index' */
        size_t index_cnt;

        /** The global radius */
        float rad;

        /** The global colour */
        unsigned char col[3];
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "BezierCurvesListDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get bezier curves list data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return core::AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return core::AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor. */
    BezierCurvesListDataCall(void);

    /** Dtor. */
    virtual ~BezierCurvesListDataCall(void);

    /**
     * Answer the number of bézier curves.
     *
     * @return The number of bézier curves
     */
    inline size_t Count(void) const {
        return this->count;
    }

    /**
     * Answer the bézier curves. Might be NULL! Do not delete the returned
     * memory.
     *
     * @return The bézier curves
     */
    inline const Curves* GetCurves(void) const {
        return this->curves;
    }

    /**
     * Answer whether or not this data has static indices.
     *
     * If this flag indicates that static index data is used, then all
     * frames will return the same number of curve lists with the same
     * lengths and exactly the same index data. Only the position, radius,
     * and colour data will differ. This allows for interpolation between
     * the frames.
     *
     * @return True if this data has static index data
     */
    inline bool HasStaticIndices(void) const {
        return this->static_indices;
    }

    /**
     * Sets the data. The object will not take ownership of the memory
     * 'curves' points to. The caller is responsible for keeping the data
     * valid as long as it is used.
     *
     * @param curves Pointer to a flat array of bézier curves.
     * @param count The number of objects stored in 'curves'
     * @param static_indices Indicates whether or not the data has static
     *                       index data
     */
    inline void SetData(const Curves* curves, size_t count, bool static_indices = false) {
        this->curves = curves;
        this->count = count;
        this->static_indices = static_indices;
    }

    /**
     * Sets the flag whether or not static index data is used.
     *
     * If this flag indicates that static index data is used, then all
     * frames will return the same number of curve lists with the same
     * lengths and exactly the same index data. Only the position, radius,
     * and colour data will differ. This allows for interpolation between
     * the frames.
     *
     * @param static_indices Set to true to indicate static index data
     */
    inline void SetHasStaticIndices(bool static_indices) {
        this->static_indices = static_indices;
    }

    /**
     * Assignment operator.
     * Makes shallow copies of all members.
     * The pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    inline BezierCurvesListDataCall& operator=(const BezierCurvesListDataCall& rhs) {
        this->curves = rhs.curves;
        this->count = rhs.count;
        this->static_indices = rhs.static_indices;
        core::AbstractGetData3DCall::operator=(rhs);
        return *this;
    }

private:
    /** The list of curves */
    const Curves* curves;

    /** The number of curves */
    size_t count;

    /**
     * Flag to indicate static index data, i.e. all frames will return the
     * same number of curve lists with the same lengths and exactly the
     * same index data. Only the position, radius, and colour data will
     * differ. This allows for interpolation between the frames.
     */
    bool static_indices;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<BezierCurvesListDataCall> BezierCurvesListDataCallDescription;


} // namespace megamol::geocalls
