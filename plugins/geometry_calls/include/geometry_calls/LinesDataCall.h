/*
 * LinesDataCall.h
 *
 * Copyright (C) 2010-2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED
#define MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED
#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/assert.h"
#include "vislib/forceinline.h"
#include "vislib/graphics/ColourRGBAu8.h"


namespace megamol::geocalls {


/**
 * Call for lines data
 */
class LinesDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Class storing a list of lines.
     * All elements are stored in flat lists without any stride!
     * The graphical primitive is GL_LINES, so the number of elements must
     * be a multiple of 2, since each line segment must explicitly store
     * it's start point and end point. In a line strip the inner points
     * must be multiplied. You can use the index array to reduce the
     * memory overhead in the colour and vertex array.
     */
    class Lines {
    public:
        /** The possible colour data types */
        enum ColourDataType {
            CDT_NONE,
            CDT_BYTE_RGB,
            CDT_BYTE_RGBA,
            CDT_FLOAT_RGB,
            CDT_FLOAT_RGBA,
            CDT_DOUBLE_RGB,
            CDT_DOUBLE_RGBA
        };

        /** Possible data types */
        enum DataType {
            DT_NONE,
            DT_BYTE, // UINT8
            DT_UINT16,
            DT_UINT32,
            DT_FLOAT,
            DT_DOUBLE
        };

        /**
         * Ctor
         */
        Lines();

        /**
         * Dtor
         */
        ~Lines();

        /**
         * Removes all data
         */
        inline void Clear() {
            this->count = 0;
            this->vrtDT = DT_NONE;
            this->vrt.dataFloat = nullptr;
            this->colDT = CDT_NONE;
            this->col.dataByte = nullptr;
            this->idxDT = DT_NONE;
            this->idx.dataByte = nullptr;
            this->globCol.Set(0, 0, 0, 0);
        }

        /**
         * Answer the number of elements. If 'IndexArray' is NULL this
         * is the number of vertex (and colour) entries stored. If
         * 'IndexArray' is not NULL this is the number of index entries
         * stored.
         *
         * @return The number of elements
         */
        inline unsigned int Count() const {
            return this->count;
        }

        /**
         * The colour data type
         *
         * @return The colour data type
         */
        inline ColourDataType ColourArrayType() const {
            return this->colDT;
        }

        /**
         * Answer the colour array. This can be NULL if the global colour
         * data should be used
         *
         * @return The colour array
         */
        inline const unsigned char* ColourArrayByte() const {
            ASSERT((this->colDT == CDT_BYTE_RGB) || (this->colDT == CDT_BYTE_RGBA));
            return this->col.dataByte;
        }

        /**
         * Answer the colour array. This can be NULL if the global colour
         * data should be used
         *
         * @return The colour array
         */
        inline const float* ColourArrayFloat() const {
            ASSERT((this->colDT == CDT_FLOAT_RGB) || (this->colDT == CDT_FLOAT_RGBA));
            return this->col.dataFloat;
        }

        /**
         * Answer the colour array. This can be NULL if the global colour
         * data should be used
         *
         * @return The colour array
         */
        inline const double* ColourArrayDouble() const {
            ASSERT((this->colDT == CDT_DOUBLE_RGB) || (this->colDT == CDT_DOUBLE_RGBA));
            return this->col.dataDouble;
        }

        /**
         * Answer the global colour value
         *
         * @return The global colour value
         */
        inline const vislib::graphics::ColourRGBAu8& GlobalColour() const {
            return this->globCol;
        }

        /**
         * Answer the ID of this line
         *
         * @return The ID
         */
        inline const size_t ID() const {
            return this->id;
        }

        /**
         * The data type of the index array
         *
         * @return The data type of the index array
         */
        inline DataType IndexArrayDataType() const {
            return this->idxDT;
        }

        /**
         * Answer the index array. This can be NULL.
         *
         * @return The index array
         */
        inline const unsigned char* IndexArrayByte() const {
            ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_BYTE));
            return this->idx.dataByte;
        }

        /**
         * Answer the index array. This can be NULL.
         *
         * @return The index array
         */
        inline const unsigned short* IndexArrayUInt16() const {
            ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_UINT16));
            return this->idx.dataUInt16;
        }

        /**
         * Answer the index array. This can be NULL.
         *
         * @return The index array
         */
        inline const unsigned int* IndexArrayUInt32() const {
            ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_UINT32));
            return this->idx.dataUInt32;
        }

        /**
         * Sets the data for this object. Ownership to all memory all
         * pointers point to will not be take by this object. The owner
         * must ensure that these pointers remain valid as long as they
         * are used. None of the pointers may be NULL; Use the proper
         * version of this method instead.
         *
         * @param cnt The number of elements
         * @param vert The vertex array (XYZ-Float)
         * @param col The global colour to be used for all lines
         */
        template<class Tp>
        inline void Set(unsigned int cnt, Tp vert, vislib::graphics::ColourRGBAu8 col) {
            ASSERT(vert != NULL);
            this->count = cnt;
            this->setVrtData(vert);
            this->colDT = CDT_NONE;
            this->col.dataByte = NULL;
            this->idxDT = DT_NONE;
            this->idx.dataByte = NULL;
            this->globCol = col;
        }

        /**
         * Sets the data for this object. Ownership to all memory all
         * pointers point to will not be take by this object. The owner
         * must ensure that these pointers remain valid as long as they
         * are used. None of the pointers may be NULL; Use the proper
         * version of this method instead.
         *
         * @param cnt The number of elements
         * @param idx The index array (UInt32)
         * @param vert The vertex array (XYZ-Float)
         * @param col The global colour to be used for all lines
         */
        template<class Tp1, class Tp2>
        inline void Set(unsigned int cnt, Tp1 idx, Tp2 vert, vislib::graphics::ColourRGBAu8 col) {
            ASSERT(idx != NULL);
            ASSERT(vert != NULL);
            this->count = cnt;
            this->setVrtData(vert);
            this->setIdxData(idx);
            this->colDT = CDT_NONE;
            this->col.dataByte = NULL;
            this->globCol = col;
        }

        /**
         * Sets the data for this object. Ownership to all memory all
         * pointers point to will not be take by this object. The owner
         * must ensure that these pointers remain valid as long as they
         * are used. None of the pointers may be NULL; Use the proper
         * version of this method instead.
         *
         * @param cnt The number of elements
         * @param vert The vertex array (XYZ-Float)
         * @param col The colour array (use same number of entries as the
         *            vertex array)
         * @param withAlpha Flag if the colour array contains RGBA(true)
         *                  or RGB(false) values
         */
        template<class Tp1, class Tp2>
        inline void Set(unsigned int cnt, Tp1 vert, Tp2 col, bool withAlpha) {
            ASSERT(vert != NULL);
            ASSERT(col != NULL);
            this->count = cnt;
            this->setVrtData(vert);
            this->setColData(col, withAlpha);
            this->idxDT = DT_NONE;
            this->idx.dataUInt32 = NULL;
            this->globCol.Set(0, 0, 0, 255);
        }

        /**
         * Sets the data for this object. Ownership to all memory all
         * pointers point to will not be take by this object. The owner
         * must ensure that these pointers remain valid as long as they
         * are used. None of the pointers may be NULL; Use the proper
         * version of this method instead.
         *
         * @param cnt The number of elements
         * @param idx The index array (UInt32)
         * @param vert The vertex array (XYZ-Float)
         * @param col The colour array (use same number of entries as the
         *            vertex array)
         * @param withAlpha Flag if the colour array contains RGBA(true)
         *                  or RGB(false) values
         */
        template<class Tp1, class Tp2, class Tp3>
        inline void Set(unsigned int cnt, Tp1 idx, Tp2 vert, Tp3 col, bool withAlpha) {
            ASSERT(idx != NULL);
            ASSERT(vert != NULL);
            ASSERT(col != NULL);
            this->count = cnt;
            this->setVrtData(vert);
            this->setColData(col, withAlpha);
            this->setIdxData(idx);
            this->globCol.Set(0, 0, 0, 255);
        }

        /**
         * Sets the list ID
         *
         * @param ID the list ID
         */
        inline void SetID(size_t ID) {
            this->id = ID;
        }

        /**
         * Answer the data type of the vertex array
         *
         * @return The data type of the vertex array
         */
        inline DataType VertexArrayDataType() const {
            return this->vrtDT;
        }

        /**
         * Answer the vertex array (XYZ-Float)
         *
         * @return The vertex array
         */
        inline const float* VertexArrayFloat() const {
            ASSERT(this->vrtDT == DT_FLOAT);
            return this->vrt.dataFloat;
        }

        /**
         * Answer the vertex array (XYZ-Double)
         *
         * @return The vertex array
         */
        inline const double* VertexArrayDouble() const {
            ASSERT(this->vrtDT == DT_DOUBLE);
            return this->vrt.dataDouble;
        }

    private:
        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(unsigned char* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_BYTE_RGBA : CDT_BYTE_RGB;
            this->col.dataByte = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(const unsigned char* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_BYTE_RGBA : CDT_BYTE_RGB;
            this->col.dataByte = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(float* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_FLOAT_RGBA : CDT_FLOAT_RGB;
            this->col.dataFloat = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(const float* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_FLOAT_RGBA : CDT_FLOAT_RGB;
            this->col.dataFloat = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(double* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_DOUBLE_RGBA : CDT_DOUBLE_RGB;
            this->col.dataDouble = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        inline void setColData(const double* data, bool withAlpha) {
            this->colDT = withAlpha ? CDT_DOUBLE_RGBA : CDT_DOUBLE_RGB;
            this->col.dataDouble = data;
        }

        /**
         * Sets the colour data
         *
         * @param the data pointer
         * @param withAlpha Flag if data contains alpha information
         */
        template<class Tp>
        inline void setColData(Tp data, bool withAlpha) {
            ASSERT(data == NULL);
            this->colDT = CDT_NONE;
            this->col.dataByte = NULL;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(unsigned char* data) {
            this->idxDT = DT_BYTE;
            this->idx.dataByte = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(const unsigned char* data) {
            this->idxDT = DT_BYTE;
            this->idx.dataByte = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(unsigned short* data) {
            this->idxDT = DT_UINT16;
            this->idx.dataUInt16 = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(const unsigned short* data) {
            this->idxDT = DT_UINT16;
            this->idx.dataUInt16 = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(unsigned int* data) {
            this->idxDT = DT_UINT32;
            this->idx.dataUInt32 = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setIdxData(const unsigned int* data) {
            this->idxDT = DT_UINT32;
            this->idx.dataUInt32 = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        template<class Tp>
        inline void setIdxData(const Tp data) {
            ASSERT(data == NULL);
            this->idxDT = DT_NONE;
            this->idx.dataUInt32 = NULL;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setVrtData(float* data) {
            this->vrtDT = DT_FLOAT;
            this->vrt.dataFloat = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setVrtData(const float* data) {
            this->vrtDT = DT_FLOAT;
            this->vrt.dataFloat = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setVrtData(double* data) {
            this->vrtDT = DT_DOUBLE;
            this->vrt.dataDouble = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        inline void setVrtData(const double* data) {
            this->vrtDT = DT_DOUBLE;
            this->vrt.dataDouble = data;
        }

        /**
         * Sets the index data
         *
         * @param the data pointer
         */
        template<class Tp>
        inline void setVrtData(Tp data) {
            ASSERT(data == NULL);
            this->vrtDT = DT_NONE;
            this->vrt.dataDouble = NULL;
        }

        /** The colour data type */
        ColourDataType colDT;

        /** The colour array */
        union _col_t {
            const unsigned char* dataByte;
            const float* dataFloat;
            const double* dataDouble;
        } col;

        /** The number of elements */
        unsigned int count;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /** The global colour */
        vislib::graphics::ColourRGBAu8 globCol;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

        /** The index array data type */
        DataType idxDT;

        /** The index array (1xunsigned int*) */
        union _idx_t {
            const unsigned char* dataByte;
            const unsigned short* dataUInt16;
            const unsigned int* dataUInt32;
        } idx;

        /** The vertex array data type */
        DataType vrtDT;

        /** The vertex array (XYZ-Float*) */
        union _vrt_t {
            const float* dataFloat;
            const double* dataDouble;
        } vrt;

        /** the line ID */
        size_t id;
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "LinesDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get lines data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
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
    LinesDataCall();

    /** Dtor. */
    ~LinesDataCall() override;

    /**
     * Answer the size of the lines lists
     *
     * @return The size of the lines lists
     */
    VISLIB_FORCEINLINE unsigned int Count() const {
        return this->count;
    }

    /**
     * Answer the lines list. Might be NULL! Do not delete the returned
     * memory.
     *
     * @return The lines list
     */
    VISLIB_FORCEINLINE const Lines* GetLines() const {
        return this->lines;
    }

    /**
     * Sets the data. The object will not take ownership of the memory
     * 'lines' points to. The caller is responsible for keeping the data
     * valid as long as it is used.
     *
     * @param count The number of lines stored in 'lines'
     * @param lines Pointer to a flat array of lines.
     * @param time The point in time for which these lines are meant.
     */
    void SetData(unsigned int count, const Lines* lines, const float time = 0.0f);

    /**
     * Sets the time the lines are called for.
     *
     * @param time The new time value.
     */
    VISLIB_FORCEINLINE void SetTime(const float time) {
        this->time = time;
    }

    /**
     * Answers the time the lines are called for.
     *
     * @return The time for which the lines are needed.
     */
    VISLIB_FORCEINLINE const float Time() const {
        return this->time;
    }


    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    LinesDataCall& operator=(const LinesDataCall& rhs);

private:
    /** The call time. */
    float time;

    /** Number of curves */
    unsigned int count;

    /** Cubic bézier curves */
    const Lines* lines;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<LinesDataCall> LinesDataCallDescription;


} // namespace megamol::geocalls

#endif /* MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED */
