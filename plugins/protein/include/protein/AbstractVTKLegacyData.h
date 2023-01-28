//
// AbstractVTKLegacyData.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINPLUGIN_ABSTRACTVTKLEGACYDATA_H_INCLUDED
#define MMPROTEINPLUGIN_ABSTRACTVTKLEGACYDATA_H_INCLUDED
#pragma once

//#include "vislib_vector_typedefs.h"
#include "vislib/String.h"

#include <cmath>

namespace megamol::protein {

/*
 * A class representing one frame of data given in the VTK legacy file format.
 */
class AbstractVTKLegacyData {

public:
    /** Enum describing the ways the data can be stored */
    enum DataEncoding { ASCII = 0, BINARY };

    /** Enum describing the possible data types */
    enum DataType {
        BIT = 0,
        UNSIGNED_CHAR,
        CHAR,
        UNSIGNED_SHORT,
        SHORT,
        UNSIGNED_INT,
        INT,
        UNSIGNED_LONG,
        LONG,
        FLOAT,
        DOUBLE
    };

    /** Enum describing the different geometry/topology the data can represent */
    enum DataGeometry { STRUCTURED_POINTS = 0, STRUCTURED_GRID, UNSTRUCTURED_GRID, POLYDATA, RECTILINEAR_GRID, FIELD };

    enum DataAssociation { POINT_DATA = 0, CELL_DATA };

    /*
     * Nested class describing a data attribute array for either point data or
     * cell data.
     */
    class AttributeArray {

    public:
        /** CTor */
        AttributeArray() : data(NULL), nElements(0), nComponents(0), allocated(0) {}

        /** Dtor */
        ~AttributeArray() {
            if (this->data) {
                delete[] this->data;
            }
        }

        /**
         * Implementation of == operator.
         *
         * @param other The data object to be compared.
         * @return 'True' if both objects are the same
         */
        bool operator==(const AttributeArray& other) const {
            return ((this->data == other.data) || (this->nElements == other.nElements) ||
                    (this->nComponents == other.nComponents) || (this->type == other.type) ||
                    (this->name == other.name) || (this->allocated == other.allocated));
        }

        /**
         * Answers a pointer to the data.
         *
         * @return The pointer to the data, might be NULL
         */
        const char* PeekData() const {
            return this->data;
        }

        /**
         * Answers the name of this data array.
         *
         * @return The name of the dat array.
         */
        vislib::StringA GetId() const {
            return this->name;
        }

        /**
         * Sets the data for this data attribute array.
         *
         * @param
         */
        void SetData(const char* data, size_t nElements, size_t nComponents, DataType type, vislib::StringA name) {

            // Compute the necessary amount of memory
            size_t nBitsPerElement;
            switch (type) {
            case BIT:
                nBitsPerElement = 1;
                break;
            case UNSIGNED_CHAR:
                nBitsPerElement = 8;
                break;
            case CHAR:
                nBitsPerElement = 8;
                break;
            case UNSIGNED_SHORT:
                nBitsPerElement = 16;
                break;
            case SHORT:
                nBitsPerElement = 16;
                break;
            case UNSIGNED_INT:
                nBitsPerElement = 32;
                break;
            case INT:
                nBitsPerElement = 32;
                break;
            case UNSIGNED_LONG:
                nBitsPerElement = 64;
                break;
            case LONG:
                nBitsPerElement = 64;
                break;
            case FLOAT:
                nBitsPerElement = 32;
                break;
            case DOUBLE:
                nBitsPerElement = 64;
                break;
            default:
                nBitsPerElement = 8;
                break;
            }
            size_t memSize = (size_t)ceil(nElements * nComponents * nBitsPerElement / 8);

            // (Re)allocate memory if necessary
            if (this->allocated < memSize) {
                if (this->data) {
                    delete[] this->data;
                }
                this->data = new char[memSize];
                this->allocated = memSize;
            }

            // Copy data
            memcpy(this->data, data, memSize);

            // Set member variables
            this->nElements = nElements;
            this->nComponents = nComponents;
            this->type = type;
            this->name = name;
        }

        /** TODO */
        size_t GetTupelCnt() const {
            return this->nElements;
        }

        /** TODO */
        size_t GetComponentCnt() const {
            return this->nComponents;
        }

    protected:
    private:
        /// The data storage pointer
        char* data;

        /// The number of elements present in the data (with each element having
        /// 'nComponent' components)
        size_t nElements;

        /// The number of components per element
        size_t nComponents;

        /// The type of the data
        DataType type;

        /// THe identifier of the data attribute
        vislib::StringA name;

        /// The currently allocated memory in bytes
        size_t allocated;
    };

    /** CTor */
    AbstractVTKLegacyData();

    /** DTor */
    virtual ~AbstractVTKLegacyData() = 0;

    /** TODO */
    static DataType GetDataTypeByString(vislib::StringA dataTypeStr) {
        if (dataTypeStr == "bit")
            return BIT;
        if (dataTypeStr == "unsigned_char")
            return UNSIGNED_CHAR;
        if (dataTypeStr == "char")
            return CHAR;
        if (dataTypeStr == "unsigned_short")
            return UNSIGNED_SHORT;
        if (dataTypeStr == "short")
            return SHORT;
        if (dataTypeStr == "unsigned_int")
            return UNSIGNED_INT;
        if (dataTypeStr == "int")
            return INT;
        if (dataTypeStr == "unsigned_long")
            return UNSIGNED_LONG;
        if (dataTypeStr == "long")
            return LONG;
        if (dataTypeStr == "float")
            return FLOAT;
        if (dataTypeStr == "double")
            return DOUBLE;
        return BIT; // TODO dirty hack to remove warnings, might change behaviour
    }

    /**
     * Get encoding of the data set.
     *
     * @return The data encoding (ascii or binary)
     */
    DataEncoding GetEncoding() const {
        return this->dataEncoding;
    }

    /**
     * Sets the data encoding of the data object.
     *
     * @param dataEncoding The data encoding of the data object.
     */
    inline void SetDataEncoding(DataEncoding dataEncoding) {
        this->dataEncoding = dataEncoding;
    }

    /**
     * Sets the data encoding of the data object.
     *
     * @param dataEncoding The data encoding of the data object.
     */
    inline void SetDataGeometry(DataGeometry dataGeometry) {
        this->dataGeometry = dataGeometry;
    }

protected:
private:
    /// The way the data is stored
    DataEncoding dataEncoding;

    /// The geometry type the data is representing
    DataGeometry dataGeometry;
};

} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_ABSTRACTVTKLEGACYDATA_H_INCLUDED
