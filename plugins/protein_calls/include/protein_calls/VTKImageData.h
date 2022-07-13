//
// VTKImageData.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 14, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCALLPLUGIN_VTKIMAGEDATA_H_INCLUDED
#define MMPROTEINCALLPLUGIN_VTKIMAGEDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#include "vislib_vector_typedefs.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"
typedef vislib::math::Cuboid<unsigned int> Cubeu;
typedef vislib::math::Vector<float, 3> Vec3f;

#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"

typedef unsigned int uint;

namespace megamol {
namespace protein_calls {

class VTKImageData {

public:
    /// Enum representing the data formats
    enum DataFormat { VTISOURCE_BINARY, VTISOURCE_ASCII, VTISOURCE_APPENDED };

    /// Enum representing different byte orderings
    enum ByteOrder { VTI_LITTLE_ENDIAN = 0, VTI_BIG_ENDIAN = 1 };

    /**
     * Nested class to represent a data array of a point data or cell data
     * object
     */
    class DataArray {
    public:
        /// Enum representing the data types
        enum DataType { VTI_UNKNOWN = 0, VTI_INT, VTI_UINT, VTI_FLOAT, VTI_DOUBLE };

        /** CTor */
        DataArray() : type(VTI_UNKNOWN), name(""), nComponents(0), data(NULL), allocated(0) {
            this->min = 0.0f;
            this->max = 0.0f;
        }

        /** DTor */
        ~DataArray() {}

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         * @return True if this and rhs are equal
         */
        bool operator==(const DataArray& rhs) const {
            return (this->type == type) && this->name.Equals(rhs.name) && (this->nComponents == rhs.nComponents) &&
                   (this->min == rhs.min) && (this->max == rhs.max) && (this->data == rhs.data) &&
                   (this->allocated == rhs.allocated);
        }

        /**
         * Answers the byte count per data element with respect to all data
         * types.
         *
         * @param t The data type.
         * @return The number of bytes for the data type.
         */
        size_t GetByteCntPerDataType(DataType t) {
            switch (t) {
            case VTI_UNKNOWN:
                return 1;
            case VTI_INT:
                return 4;
            case VTI_UINT:
                return 4;
            case VTI_FLOAT:
                return 4;
            case VTI_DOUBLE:
                return 8;
            default:
                return 1;
            }
        }

        /**
         * Answers a pointer to the actual data chunk.
         *
         * @return A pointer to the actual data chunk.
         */
        inline char* PeekData() const {
            return this->data;
        }

        /**
         * Updates the data stored in the array. Allocates memory if necessary.
         *
         * @param data        The data to be stored.
         * @param min         The minimum data value.
         * @param max         The maximum data value.
         * @param t           The data's data type.
         * @param id          The name of the data array.
         * @param nComponents The number of components per data element.
         */
        void UpdateData(
            const char* data, double min, double max, DataArray::DataType t, vislib::StringA id, size_t nComponents) {
            //
            //            printf("Update data\n");
            //            printf("nComponents %u\n", nComponents);


            this->type = t;
            this->name = id;
            this->min = min;
            this->max = max;
            this->nComponents = nComponents;

            uint gridSize = (this->extent.Width() + 1) * (this->extent.Depth() + 1) * (this->extent.Height() + 1);

            //            printf("nComponents %u\n", gridSize);

            size_t nBytesPerElement = nComponents * this->GetByteCntPerDataType(t);

            //            printf("needed size %u\n", gridSize*nBytesPerElement);
            //            printf("allocated %u\n", allocated);

            // Check whether the memory needs to be (re-)allocated
            if (this->allocated < gridSize * nBytesPerElement) {
                //                printf("allocate new\n");
                if (this->data != NULL) {
                    //                    printf("delete old\n");
                    delete[] this->data;
                }
                this->data = new char[gridSize * nBytesPerElement];
                this->allocated = gridSize * nBytesPerElement;
            }

            // Copy data
            memcpy(this->data, data, this->allocated);
        }

        /**
         * Sets the extent of the data.
         *
         * @param extent The extent of the data.
         */
        inline void SetExtent(vislib::math::Cuboid<uint> extent) {
            this->extent = extent;
        }

        /**
         * Answers the minimum data value.
         *
         * @return The minimum data value.
         */
        inline double GetMin() const {
            return this->min;
        }

        /**
         * Answers the maximum data value.
         *
         * @return The maximum data value.
         */
        inline double GetMax() const {
            return this->max;
        }

        /**
         * Answers the number of elements in the array.
         *
         * @return The number of elements in the array.
         */
        inline size_t GetSize() const {
            return (this->extent.Width() + 1) * (this->extent.Height() + 1) * (this->extent.Depth() + 1);
        }

        /**
         * Answers the name of the data array.
         *
         * @return The name of the data array.
         */
        inline vislib::StringA GetId() const {
            return this->name;
        }

        /**
         * Answers the type of the data array.
         *
         * @return The type of the data array.
         */
        inline DataType GetType() const {
            return this->type;
        }

        /**
         * Deallocates memory and resets all parameters to their initial state.
         */
        inline void Release() {
            if (this->data != NULL) {
                delete[] this->data;
            }
            this->data = NULL;
            this->type = VTI_UNKNOWN;
            this->name = "";
            this->data = NULL;
            this->nComponents = 0;
            this->min = 0.0;
            this->max = 0.0;
            this->allocated = 0;
        }

        /**
         * Answers the number of components of each element in the data array.
         *
         * @return The number of components of each element.
         */
        inline size_t GetNumberOfComponents() const {
            return this->nComponents;
        }

    private:
        DataType type;                     ///> The data type used in this data array
        vislib::StringA name;              ///> The id of the data array
        size_t nComponents;                ///> The number of components per data element
        double min, max;                   ///> The range of the data
        vislib::math::Cuboid<uint> extent; ///> The piece's extent

        char* data;       ///> The actual data
        size_t allocated; ///> The current ammount of allocated memory
    };

    /**
     * Nested class to represent a pieces cell data
     */
    class CellData {

    public:
        /** CTor */
        CellData() : extent(0, 0, 0, 0, 0, 0) {
            this->dataArrays.SetCount(0);
        }

        /** DTor */
        ~CellData() {
            this->Release();
        }

        /**
         * Answers the number of elements in a specified data array.
         *
         * @param dataIdx The index of the specified array.
         * @return The number of elements in the array.
         */
        inline size_t GetArraySize(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetSize();
        }

        /**
         * Answers the number of data arrays in this data object.
         *
         * @return The number of data arrays.
         */
        size_t GetDataArrayCount() const {
            return this->dataArrays.Count();
        }

        /**
         * Answers the minimum value of the specified data array.
         *
         * @param The index of the data array.
         * @return The minimum value.
         */
        double GetDataArrayMin(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetMin();
        }

        /**
         * Answers the maximum value of the specified data array.
         *
         * @param The index of the data array.
         * @return The maximum value.
         */
        double GetDataArrayMax(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetMax();
        }

        /**
         * Returns the data array with the index idx.
         *
         * @param idx The data arrays index.
         * @return The data requested array or NULL.
         */
        const DataArray* GetDataArray(unsigned int idx) const {
            using namespace megamol::core::utility::log;
            if (idx >= this->dataArrays.Count()) {
                Log::DefaultLog.WriteError("CellData: Requested idx out of bound, returning NULL.");
                return NULL;
            } else {
                return this->dataArrays[idx];
            }
        }

        /**
         * Returns the data array with the name 'id'
         *
         * @param idx The data arrays index.
         * @return The data requested array or NULL.
         */
        const DataArray* GetDataArray(vislib::StringA id) const {
            using namespace megamol::core::utility::log;

            // Check whether the id is in use
            bool isUsed = false;
            int idx = -1;
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                if (this->dataArrays[i]->GetId() == id) {
                    isUsed = true;
                    idx = i;
                    break;
                }
            }

            // If the id is not in use: return null
            if (!isUsed) {
                Log::DefaultLog.WriteError("CellData: Requested id '%s' not in use, returning NULL.", id.PeekBuffer());
                return NULL;
            } else { // else: return the data array
                return this->dataArrays[idx];
            }
        }

        /**
         * Creates or updates a data array wih the name "id".
         *
         * @param id          The name of the data array.
         * @param data        The actual data chunk.
         * @param min         The minimum data value.
         * @param double      The maximum data value.
         * @param t           The data type.
         * @param nComponents The number of components per element.
         */
        void UpdateDataArray(
            vislib::StringA id, const char* data, double min, double max, DataArray::DataType t, size_t nComponents) {

            // Check whether the id is already in use
            bool isUsed = false;
            int idx = -1;
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                if (this->dataArrays[i]->GetId() == id) {
                    isUsed = true;
                    idx = i;
                    break;
                }
            }

            // If the id is already in use: update data
            if (isUsed) {
                this->dataArrays[idx]->UpdateData(data, min, max, t, id, nComponents);
            } else { // else: create new data array
                this->dataArrays.Append(new DataArray());
                this->dataArrays.Last()->SetExtent(this->extent);
                this->dataArrays.Last()->UpdateData(data, min, max, t, id, nComponents);
            }
        }

        /**
         * Sets the piece's extent.
         *
         * @param extent The piece's extent.
         */
        inline void SetExtent(vislib::math::Cuboid<uint> extent) {
            this->extent = extent;
        }

        /**
         * Deallocates all memory.
         */
        inline void Release() {
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                delete this->dataArrays[i];
            }
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         * @return True if this and rhs are equal
         */
        bool operator==(const CellData& rhs) const {
            return (this->dataArrays == rhs.dataArrays) && (this->extent == rhs.extent);
        }

        /**
         * Answers the number of data arrays.
         *
         * @return The number of data arrays.
         */
        unsigned int GetArrayCnt() {
            return (unsigned int)this->dataArrays.Count();
        }

        /**
         * Answers the number of components of each element in the data array
         * with the index 'dataIdx'.
         *
         * @return The number of components of each element.
         */
        inline size_t GetArrayNumberOfComponents(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetNumberOfComponents();
        }

        /**
         * Answers the name of the data array with the index 'dataIdx'.
         *
         * @return The name of the data array.
         */
        inline vislib::StringA GetArrayId(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetId();
        }

        /**
         * Answers the data type of the data array with the index 'dataIdx'.
         *
         * @return The type of the data array.
         */
        inline DataArray::DataType GetArrayType(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetType();
        }

    private:
        /// The list of data arrays
        vislib::Array<DataArray*> dataArrays;

        /// The piece's extent
        vislib::math::Cuboid<uint> extent;
    };

    /**
     * Nested class to represent a pieces point data
     */
    class PointData {
    public:
        /** CTor */
        PointData() : extent(0, 0, 0, 0, 0, 0) {
            this->dataArrays.SetCount(0);
        }

        /** DTor */
        ~PointData() {
            this->Release();
        }

        /**
         * Answers the number of elements in a specified data array.
         *
         * @param dataIdx The index of the specified array.
         * @return The number of elements in the array.
         */
        inline size_t GetArraySize(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetSize();
        }

        /**
         * Answers the minimum value of the specified data array.
         *
         * @param The index of the data array.
         * @return The minimum value.
         */
        double GetDataArrayMin(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetMin();
        }

        /**
         * Answers the maximum value of the specified data array.
         *
         * @param The index of the data array.
         * @return The maximum value.
         */
        double GetDataArrayMax(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetMax();
        }

        /**
         * Returns the data array with the index idx.
         *
         * @param idx The data arrays index.
         * @return The data requested array or NULL.
         */
        const DataArray* GetDataArray(unsigned int idx) const {
            using namespace megamol::core::utility::log;

            if (idx >= this->dataArrays.Count()) {
                Log::DefaultLog.WriteError("PointData: Requested idx out of bound, returning NULL.");
                return NULL;
            } else {
                return this->dataArrays[idx];
            }
        }

        /**
         * Answers the number of data arrays in this data object.
         *
         * @return The number of data arrays.
         */
        size_t GetDataArrayCount() const {
            return this->dataArrays.Count();
        }

        /**
         * Answers the name of the data array with the index 'dataIdx'.
         *
         * @return The name of the data array.
         */
        inline vislib::StringA GetArrayId(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetId();
        }

        /**
         * Answers the data type of the data array with the index 'dataIdx'.
         *
         * @return The type of the data array.
         */
        inline DataArray::DataType GetArrayType(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetType();
        }

        /**
         * Returns the data array with the name 'id'
         *
         * @param idx The data arrays index.
         * @return The data requested array or NULL.
         */
        const DataArray* GetDataArray(vislib::StringA id) const {
            using namespace megamol::core::utility::log;

            // Check whether the id is in use
            bool isUsed = false;
            int idx = -1;
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                if (this->dataArrays[i]->GetId() == id) {
                    isUsed = true;
                    idx = i;
                    break;
                }
            }

            // If the id is not in use: return null
            if (!isUsed) {
                Log::DefaultLog.WriteError("PointData: Requested id '%s' not in use, returning NULL.", id.PeekBuffer());
                return NULL;
            } else { // else: return the data array
                return this->dataArrays[idx];
            }
        }

        /**
         * Creates or updates a data array wih the name "id".
         *
         * @param id          The name of the data array.
         * @param data        The actual data chunk.
         * @param min         The minimum data value.
         * @param double      The maximum data value.
         * @param t           The data type.
         * @param nComponents The number of components per element.
         */
        void UpdateDataArray(
            vislib::StringA id, const char* data, double min, double max, DataArray::DataType t, size_t nComponents) {

            // Check whether the id is already in use
            bool isUsed = false;
            int idx = -1;
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                if (this->dataArrays[i]->GetId() == id) {
                    isUsed = true;
                    idx = i;
                    break;
                }
            }

            // If the id is already in use: update data
            if (isUsed) {
                this->dataArrays[idx]->UpdateData(data, min, max, t, id, nComponents);
            } else { // else: create new data array
                this->dataArrays.Append(new DataArray());
                this->dataArrays.Last()->SetExtent(this->extent);
                this->dataArrays.Last()->UpdateData(data, min, max, t, id, nComponents);
            }
        }

        /**
         * Sets the piece's extent.
         *
         * @param extent The piece's extent.
         */
        inline void SetExtent(vislib::math::Cuboid<uint> extent) {
            this->extent = extent;
        }

        /**
         * Deallocates all memory.
         */
        inline void Release() {
            for (unsigned int i = 0; i < this->dataArrays.Count(); ++i) {
                this->dataArrays[i]->Release();
            }
            this->dataArrays.Clear();
            this->dataArrays.SetCount(0);
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         * @return True if this and rhs are equal
         */
        bool operator==(const PointData& rhs) const {
            return (this->dataArrays == rhs.dataArrays) && (this->extent == rhs.extent);
        }

        /**
         * Answers the number of components of each element in the data array
         * with the index 'dataIdx'.
         *
         * @return The number of components of each element.
         */
        inline size_t GetArrayNumberOfComponents(unsigned int dataIdx) const {
            return this->dataArrays[dataIdx]->GetNumberOfComponents();
        }

    private:
        /// The list of data arrays
        vislib::Array<DataArray*> dataArrays;

        /// The piece's extent
        vislib::math::Cuboid<uint> extent;
    };

    /**
     * Nested class representing one piece of the data
     */
    class Piece {
    public:
        /**
         * Answers the number of data arrays in the point data.
         *
         * @return The number of data arrays in the point data.
         */
        size_t GetArrayCntPointData() const {
            return this->pointData.GetDataArrayCount();
        }

        /**
         * Answers the number of elements in a specified point data array.
         *
         * @param dataIdx The index of the specified array.
         * @return The number of elements in the array.
         */
        inline size_t GetPointArraySize(unsigned int dataIdx) const {
            return this->pointData.GetArraySize(dataIdx);
        }

        /**
         * Answers the number of elements in a specified cell data array.
         *
         * @param dataIdx The index of the specified array.
         * @return The number of elements in the array.
         */
        inline size_t GetCellArraySize(unsigned int dataIdx) const {
            return this->cellData.GetArraySize(dataIdx);
        }

        /**
         * Answers the number of data arrays in the point data.
         *
         * @return The number of data arrays in the point data.
         */
        size_t GetArrayCntCellData() const {
            return this->cellData.GetDataArrayCount();
        }

        /**
         * Answers the minimum point data value of the specified data array.
         *
         * @param The index of the data array.
         * @return The minimum value.
         */
        double GetPointDataArrayMin(unsigned int dataIdx) const {
            return this->pointData.GetDataArrayMin(dataIdx);
        }

        /**
         * Answers the maximum point data value of the specified data array.
         *
         * @param The index of the data array.
         * @return The maximum value.
         */
        double GetPointDataArrayMax(unsigned int dataIdx) const {
            return this->pointData.GetDataArrayMax(dataIdx);
        }

        /**
         * Answers the minimum cell data value of the specified data array.
         *
         * @param The index of the data array.
         * @return The minimum value.
         */
        double GetCellDataArrayMin(unsigned int dataIdx) const {
            return this->cellData.GetDataArrayMin(dataIdx);
        }

        /**
         * Answers the maximum cell data value of the specified data array.
         *
         * @param The index of the data array.
         * @return The maximum value.
         */
        double GetCellDataArrayMax(unsigned int dataIdx) const {
            return this->cellData.GetDataArrayMax(dataIdx);
        }

        /**
         * Answers the data type of the data array with the index 'dataIdx' of
         * the cell data.
         *
         * @return The type of the data array.
         */
        inline DataArray::DataType GetCellArrayType(unsigned int dataIdx) const {
            return this->cellData.GetArrayType(dataIdx);
        }

        /**
         * Answers the data type of the data array with the index 'dataIdx' of
         * the point data.
         *
         * @return The type of the data array.
         */
        inline DataArray::DataType GetPointArrayType(unsigned int dataIdx) const {
            return this->pointData.GetArrayType(dataIdx);
        }

        /** CTor */
        Piece() : extent(0, 0, 0, 0, 0, 0) {}

        /** Dtor */
        ~Piece() {}

        /**
         * Adds a data array to the point data or updates a current one.
         *
         * @param data        The actual data.
         * @param min         The minimum data value.
         * @param max         The maximum data value.
         * @param t           The data type of the data array.
         * @param id          The name of the data array (has to be unique).
         * @param nComponents The number of components for each element.
         */
        void SetPointData(
            const char* data, double min, double max, DataArray::DataType t, vislib::StringA id, size_t nComponents) {
            this->pointData.UpdateDataArray(id, data, min, max, t, nComponents);
        }

        /**
         * Adds a data array to the cell data or updates a current one.
         *
         * @param data        The actual data.
         * @param min         The minimum data value.
         * @param max         The maximum data value.
         * @param t           The data type of the data array.
         * @param id          The name of the data array (has to be unique).
         * @param nComponents The number of components for each element.
         */
        void SetCellData(
            const char* data, double min, double max, DataArray::DataType t, vislib::StringA id, size_t nComponents) {
            this->cellData.UpdateDataArray(id, data, min, max, t, nComponents);
        }

        /**
         * Get point data array according to the array's id. Might return NULL
         * if the requested id is not in use.
         *
         * @param The arrays id.
         */
        const DataArray* PeekPointData(vislib::StringA id) const {
            return this->pointData.GetDataArray(id);
        }

        /**
         * Get point data array according to the array's index. Might return NULL
         * if the requested id is not in use.
         *
         * @param The arrays id.
         */
        const DataArray* PeekPointData(unsigned int arrayIdx) const {
            return this->pointData.GetDataArray(arrayIdx);
        }

        /**
         * Get cell data array according to the array's id. Might return NULL
         * if the requested id is not in use.
         *
         * @param The arrays id.
         */
        const DataArray* PeekCellData(vislib::StringA id) const {
            return this->cellData.GetDataArray(id);
        }

        /**
         * Get cell data array according to the array's id. Might return NULL
         * if the requested id is not in use.
         *
         * @param The arrays id.
         */
        const DataArray* PeekCellData(unsigned int arrayIdx) const {
            return this->cellData.GetDataArray(arrayIdx);
        }

        /**
         * Release all allocated memory.
         */
        inline void Release() {
            this->pointData.Release();
            this->cellData.Release();
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         * @return True if this and rhs are equal
         */
        bool operator==(const Piece& rhs) const {
            return (this->pointData == rhs.pointData) && (this->cellData == rhs.cellData) &&
                   (this->extent == rhs.extent);
        }

        /**
         * Answers the piece's extent.
         *
         * @return The piece's extent.
         */
        Cubeu GetExtent() const {
            return this->extent;
        }

        /**
         * Sets the piece's extent.
         *
         * @param The piece's extent.
         */
        inline void SetExtent(vislib::math::Cuboid<uint> extent) {
            this->extent = extent;
            this->cellData.SetExtent(vislib::math::Cuboid<uint>(extent.Left(), extent.Bottom(), extent.Back(),
                extent.Right() - 1, extent.Top() - 1, extent.Front() - 1));
            this->pointData.SetExtent(extent);
        }

        /**
         * Answers the number of components of each element in the data array
         * with the index 'dataIdx' of the pieces point data.
         *
         * @return The number of components of each element.
         */
        inline size_t GetPointDataArrayNumberOfComponents(unsigned int dataIdx) const {
            return this->pointData.GetArrayNumberOfComponents(dataIdx);
        }

        /**
         * Answers the number of components of each element in the data array
         * with the index 'dataIdx' of the pieces cell data.
         *
         * @return The number of components of each element.
         */
        inline size_t GetCellDataArrayNumberOfComponents(unsigned int dataIdx) const {
            return this->cellData.GetArrayNumberOfComponents(dataIdx);
        }

        /**
         * Answers the name of the cell data array with the index 'dataIdx'.
         *
         * @return The name of the cell data array.
         */
        inline vislib::StringA GetCellDataArrayId(unsigned int dataIdx) const {
            return this->cellData.GetArrayId(dataIdx);
        }

        /**
         * Answers the name of the point data array with the index 'dataIdx'.
         *
         * @return The name of the point data array.
         */
        inline vislib::StringA GetPointDataArrayId(unsigned int dataIdx) const {
            return this->pointData.GetArrayId(dataIdx);
        }

    private:
        CellData cellData;                 ///> The piece's cell data
        PointData pointData;               ///> The piece's point data
        vislib::math::Cuboid<uint> extent; ///> The piece's extent
    };

    /** CTor */
    VTKImageData() : wholeExtent(0, 0, 0, 0, 0, 0), origin(0.0f, 0.0f, 0.0f), spacing(0.0f, 0.0f, 0.0f) {

        this->pieces.SetCount(0);
    }

    /** DTor */
    ~VTKImageData() {
        this->Release();
    }

    /**
     * Answers the number of elements in a specified point data array.
     *
     * @param dataIdx The index of the specified array.
     * @param pieceIdx The piece's index.
     * @return The number of elements in the array.
     */
    inline size_t GetPiecePointArraySize(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointArraySize(dataIdx);
    }

    /**
     * Answers the number of elements in a specified cell data array.
     *
     * @param dataIdx The index of the specified array.
     * @param pieceIdx The piece's index.
     * @return The number of elements in the array.
     */
    inline size_t GetPieceCellArraySize(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellArraySize(dataIdx);
    }

    /**
     * Answers the data set's whole extent.
     *
     * @return The whole extent.
     */
    vislib::math::Cuboid<uint> GetWholeExtent() const {
        return this->wholeExtent;
    }

    /**
     * Answers the data set's origin.
     *
     * @return The data set's origin
     */
    Vec3f GetOrigin() const {
        return this->origin;
    }

    /**
     * Answers the lattice spacing for the data set.
     *
     * @return The data set's spacing.
     */
    Vec3f GetSpacing() const {
        return this->spacing;
    }

    /**
     * Answers the minimum point data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The minimum value.
     */
    double GetPointDataArrayMin(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointDataArrayMin(dataIdx);
    }

    /**
     * Answers the maximum point data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The maximum value.
     */
    double GetPointDataArrayMax(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointDataArrayMax(dataIdx);
    }

    /**
     * Answers the minimum cell data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The minimum value.
     */
    double GetCellDataArrayMin(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellDataArrayMin(dataIdx);
    }

    /**
     * Answers the maximum cell data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The maximum value.
     */
    double GetCellDataArrayMax(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellDataArrayMax(dataIdx);
    }

    /**
     * Sets the data set's whole extent.
     *
     * @param wholeExtent The data set's whole extent.
     */
    inline void SetWholeExtent(vislib::math::Cuboid<uint> wholeExtent) {
        this->wholeExtent = wholeExtent;
    }

    /**
     * Sets the data set's origin.
     *
     * @param origin The data set's origin.
     */
    inline void SetOrigin(Vec3f origin) {
        this->origin = origin;
    }

    /**
     * Sets the data sets spacing.
     *
     * @param spacing The lattice spacing
     */
    inline void SetSpacing(Vec3f spacing) {
        this->spacing = spacing;
    }

    /**
     * Sets the number of pieces.
     *
     * @param The number of pieces.
     */
    inline void SetNumberOfPieces(uint nPieces) {
        this->pieces.SetCount(nPieces);
    }

    /**
     * Sets the number of pieces.
     *
     * @param The number of pieces.
     */
    void SetPieceExtent(unsigned int pieceIdx, vislib::math::Cuboid<unsigned int> extent) {
        using namespace megamol::core::utility::log;
        if (pieceIdx >= this->pieces.Count()) {
            Log::DefaultLog.WriteError("VTKImageData: Could not set extent of piece #%u (number of pieces is %u).",
                pieceIdx, this->pieces.Count());
            return;
        }
        this->pieces[pieceIdx].SetExtent(extent);
    }

    /**
     * Answers the extent of a piece.
     *
     * @return The extent of the requested piece.
     */
    Cubeu GetPieceExtent(unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetExtent();
    }

    /**
     * Answers the number of pieces.
     *
     * @return The number of pieces.
     */
    inline unsigned int GetNumberOfPieces() const {
        return static_cast<unsigned int>(this->pieces.Count());
    }

    /**
     * Answers the number of data arrays present in the point data of the piece
     * with the index 'pieceIdx'.
     *
     * @param pieceIdx The index of the piece.
     * @return The number of data arrays in the specified piece.
     */
    inline size_t GetArrayCntOfPiecePointData(unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetArrayCntPointData();
    }

    /**
     * Answers the number of data arrays present in the cell data present in the
     * piece with the index 'pieceIdx'.
     *
     * @param pieceIdx The index of the piece.
     * @return The number of data arrays in the specified piece.
     */
    inline size_t GetArrayCntOfPieceCellData(unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetArrayCntCellData();
    }

    /**
     * Get point data array according to the array's id. Might return NULL
     * if the requested id is not in use.
     *
     * @param id The arrays id.
     * @param pieceIdx The piece's idx.
     * @return The data array.
     */
    const DataArray* PeekPointData(vislib::StringA id, unsigned int pieceIdx) const {
        using namespace megamol::core::utility::log;
        if (pieceIdx >= this->pieces.Count()) {
            Log::DefaultLog.WriteError("VTKImageData: Piece index out of bounds, returning NULL.");
        }
        return this->pieces[pieceIdx].PeekPointData(id);
    }

    /**
     * Get point data array according to the array's index. Might return NULL
     * if the requested id is not in use.
     *
     * @param id The arrays id.
     * @param pieceIdx The piece's idx.
     * @return The data array.
     */
    const DataArray* PeekPointData(unsigned int arrayIdx, unsigned int pieceIdx) const {
        using namespace megamol::core::utility::log;
        if (pieceIdx >= this->pieces.Count()) {
            Log::DefaultLog.WriteError("VTKImageData: Piece index out of bounds, returning NULL.");
        }
        return this->pieces[pieceIdx].PeekPointData(arrayIdx);
    }

    /**
     * Get cell data array according to the array's id. Might return NULL
     * if the requested id is not in use.
     *
     * @param id The arrays id.
     * @param pieceIdx The piece's idx.
     * @return The data array.
     */
    const DataArray* PeekCellData(vislib::StringA id, unsigned int pieceIdx) const {
        using namespace megamol::core::utility::log;
        if (pieceIdx >= this->pieces.Count()) {
            Log::DefaultLog.WriteError("VTKImageData: Piece index out of bounds, returning NULL.");
        }
        return this->pieces[pieceIdx].PeekCellData(id);
    }

    /**
     * Get cell data array according to the array's index. Might return NULL
     * if the requested id is not in use.
     *
     * @param id The arrays id.
     * @param pieceIdx The piece's idx.
     * @return The data array.
     */
    const DataArray* PeekCellData(unsigned int arrayIdx, unsigned int pieceIdx) const {
        using namespace megamol::core::utility::log;
        if (pieceIdx >= this->pieces.Count()) {
            Log::DefaultLog.WriteError("VTKImageData: Piece index out of bounds, returning NULL.");
        }
        return this->pieces[pieceIdx].PeekCellData(arrayIdx);
    }

    /**
     * Adds a data array to the point data or updates a current one.
     *
     * @param data        The actual data.
     * @param min         The minimum data value.
     * @param max         The maximum data value.
     * @param t           The data type of the data array.
     * @param id          The name of the data array (has to be unique).
     * @param nComponents The number of components for each element.
     * @param pieceIdx    The index of the piece.
     */
    void SetPointData(const char* data, double min, double max, DataArray::DataType t, vislib::StringA id,
        size_t nComponents, unsigned int pieceIdx) {
        this->pieces[pieceIdx].SetPointData(data, min, max, t, id, nComponents);
    }

    /**
     * Adds a data array to the cell data or updates a current one.
     *
     * @param data        The actual data.
     * @param min         The minimum data value.
     * @param max         The maximum data value.
     * @param t           The data type of the data array.
     * @param id          The name of the data array (has to be unique).
     * @param nComponents The number of components for each element.
     * @param pieceIdx    The index of the piece.
     */
    void SetCellData(const char* data, double min, double max, DataArray::DataType t, vislib::StringA id,
        size_t nComponents, unsigned int pieceIdx) {
        this->pieces[pieceIdx].SetCellData(data, min, max, t, id, nComponents);
    }

    /**
     * Releases all data arrays of all pieces.
     */
    void Release() {
        for (unsigned int i = 0; i < this->pieces.Count(); ++i) {
            this->pieces[i].Release();
        }
    }

    /**
     * Answers the number of components of each element in the data array
     * with the index 'dataIdx' of the point data of the piece with the index
     * 'pieceIdx'.
     *
     * @param dataIdx  The data array's index.
     * @param pieceIdx The piece's index.
     * @return The number of components of each element.
     */
    inline size_t GetPointDataArrayNumberOfComponents(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointDataArrayNumberOfComponents(dataIdx);
    }

    /**
     * Answers the number of components of each element in the data array
     * with the index 'dataIdx' of the cell data of the piece with the index
     * 'pieceIdx'.
     *
     * @param dataIdx  The data array's index.
     * @param pieceIdx The piece's index.
     * @return The number of components of each element.
     */
    inline size_t GetCellDataArrayNumberOfComponents(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellDataArrayNumberOfComponents(dataIdx);
    }

    /**
     * Answers the name of the cell data array with the index 'dataIdx' of the
     * piece with the piece index 'pieceIdx'.
     *
     * @return The name of the cell data array.
     */
    inline vislib::StringA GetCellDataArrayId(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellDataArrayId(dataIdx);
    }

    /**
     * Answers the name of the point data array with the index 'dataIdx' of the
     * piece with the piece index 'pieceIdx'.
     *
     * @return The name of the point data array.
     */
    inline vislib::StringA GetPointDataArrayId(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointDataArrayId(dataIdx);
    }

    /**
     * Answers the data type of the data array with the index 'dataIdx' of
     * the point data of the piece 'pieceIdx'.
     *
     * @return The type of the data array.
     */
    inline DataArray::DataType GetPiecePointArrayType(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetPointArrayType(dataIdx);
    }

    /**
     * Answers the data type of the data array with the index 'dataIdx' of
     * the cell data of the piece 'pieceIdx'.
     *
     * @return The type of the data array.
     */
    inline DataArray::DataType GetPieceCellArrayType(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->pieces[pieceIdx].GetCellArrayType(dataIdx);
    }

protected:
private:
    /// The data sets number of elements in each direction
    vislib::math::Cuboid<uint> wholeExtent;

    /// The data sets origin in world space coordinates
    Vec3f origin;

    /// The data sets spacing in each direction
    Vec3f spacing;

    vislib::Array<Piece> pieces; ///> The array of pieces of this data set
};

} // namespace protein_calls
} // namespace megamol

#endif // MMPROTEINCALLPLUGIN_VTKIMAGEDATA_H_INCLUDED
