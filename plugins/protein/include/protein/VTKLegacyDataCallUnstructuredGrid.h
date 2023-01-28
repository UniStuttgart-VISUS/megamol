//
// VTKLegacyDataCallUnstructuredGrid.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINPLUGIN_VTKDATACALLUNSTRUCTUREDGRID_H_INCLUDED
#define MMPROTEINPLUGIN_VTKDATACALLUNSTRUCTUREDGRID_H_INCLUDED
#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "protein/VTKLegacyDataUnstructuredGrid.h"
#include "protein_calls/Interpol.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"

namespace megamol::protein {

class VTKLegacyDataCallUnstructuredGrid : public core::AbstractGetData3DCall {

public:
    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /** Ctor. */
    VTKLegacyDataCallUnstructuredGrid();

    /** Dtor. */
    ~VTKLegacyDataCallUnstructuredGrid() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VTKLegacyDataCallUnstructuredGrid";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call for VTK unstructured grid data.";
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
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return core::AbstractGetData3DCall::FunctionName(idx);
    }

    /**
     * Answer the call time
     *
     * @return the call time
     */
    float GetCalltime() const {
        return this->calltime;
    }

    /**
     * Sets the call time to request data for.
     *
     * @param call time The call time to request data for.
     */
    void SetCalltime(float calltime) {
        this->calltime = calltime;
    }

    /**
     * Sets the pointer to the data.
     *
     * @param Pointer to the data.
     */
    void SetData(const VTKLegacyDataUnstructuredGrid* data) {
        this->data = data;
    }

    /** TODO */
    const float* PeekPoints() const {
        return this->data->PeekPoints();
    }

    /** TODO */
    const int* PeekCells() const {
        return this->data->PeekCells();
    }

    /** TODO */
    inline size_t GetPointDataCount() const {
        return this->data->GetPointDataCount();
    }

    /**
     * Answers the pointer to the cell types
     *
     * @return The pointer to the data, or NULL if the array is not there
     */
    inline const VTKLegacyDataUnstructuredGrid::CellType* PeekCellTypes() const {
        //        // DEBUG Print vertex positions
        //        for (size_t p = 0; p < this->nPoints; ++p) {
        //            printf("!!%i: (%f %f %f)\n", p,
        //                    this->points[3*p+0],
        //                    this->points[3*p+1],
        //                    this->points[3*p+2]);
        //        }
        //        // END DEBUG
        return this->data->PeekCellTypes();
    }

    /**
     * Get point data array according to the array's id and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @return A pointer to the requested data array or NULL.
     */
    const AbstractVTKLegacyData::AttributeArray* PeekPointDataByName(vislib::StringA id) const {
        return this->data->PeekPointDataByName(id);
    }

    /**
     * Get point data array according to the array's index and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @return A pointer to the requested data array or NULL.
     */
    const AbstractVTKLegacyData::AttributeArray* PeekPointDataByIndex(unsigned int arrayIdx) const {
        return this->data->PeekPointDataByIndex(arrayIdx);
    }

    /**
     * Get cell data array according to the array's id and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetCellDataByName(vislib::StringA id) const {
        return this->data->PeekCellDataByName(id);
    }

    /**
     * Get cell data array according to the array's index and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetCellDataByIdx(unsigned int arrayIdx) const {
        return this->data->PeekCellDataByIndex(arrayIdx);
    }

    /**
     * Answers the number of data arrays present in the point data.
     *
     * @return The number of data arrays in the specified piece.
     */
    //    inline size_t GetArrayCntOfPointData() const {
    //        return this->data->GetArrayCntOfPointData();
    //    }

    /**
     * Answers the number of data arrays present in the cell data.
     *
     * @return The number of data arrays in the specified piece.
     */
    //    inline size_t GetArrayCntOfCellData() const {
    //        return this->data->GetArrayCntOfCellData();
    //    }

    /**
     * Answers the number of components of each element in the data array
     * with the index 'dataIdx' of the point data of the piece with the index
     * 'pieceIdx'.
     *
     * @param dataIdx  The data array's index.
     * @return The number of components of each element.
     */
    //    inline size_t GetPointDataArrayNumberOfComponents(unsigned int dataIdx) const {
    //        return this->data->GetPointDataArrayNumberOfComponents(dataIdx);
    //    }

    /**
     * Answers the number of components of each element in the data array
     * with the index 'dataIdx'.
     *
     * @param dataIdx  The data array's index.
     * @param pieceIdx The piece's index.
     * @return The number of components of each element.
     */
    //    inline size_t GetCellDataArrayNumberOfComponents(unsigned int dataIdx) const {
    //        return this->data->GetCellDataArrayNumberOfComponents(dataIdx);
    //    }

    /**
     * Answers the name of the cell data array with the index 'dataIdx'.
     *
     * @return The name of the cell data array.
     */
    //    inline vislib::StringA GetCellDataArrayId(unsigned int dataIdx,
    //            unsigned int pieceIdx) const {
    //        return this->data->GetCellDataArrayId(dataIdx);
    //    }

    /**
     * Answers the name of the point data array with the index 'dataIdx' of the
     * piece with the piece index 'pieceIdx'.
     *
     * @return The name of the point data array.
     */
    //    inline vislib::StringA GetPointDataArrayId(unsigned int dataIdx) const {
    //        return this->data->GetPointDataArrayId(dataIdx);
    //    }

    /**
     * Answers the number of elements in a specified point data array.
     *
     * @param dataIdx The index of the specified array.
     * @return The number of elements in the array.
     */
    //    inline size_t GetPointDataArraySize(unsigned int dataIdx) const {
    //        return this->data->GetPointArraySize(dataIdx);
    //    }

    /**
     * Answers the number of elements in a specified cell data array.
     *
     * @param dataIdx The index of the specified array.
     * @return The number of elements in the array.
     */
    //    inline size_t GetCellDataArraySize(unsigned int dataIdx) const {
    //        return this->data->GetCellArraySize(dataIdx);
    //    }

    /** TODO */
    size_t GetNumberOfPoints() const {
        return this->data->GetNumberOfPoints();
    }

    /** TODO */
    size_t GetCellDataSize() const {
        return this->data->GetCellDataSize();
    }

    /** TODO */
    size_t GetNumberOfCells() const {
        return this->data->GetNumberOfCells();
    }

private:
    float calltime;  ///> The exact requested/stored call time
    size_t frameCnt; ///> The number of frames

    /// The current frame's data (pointer)
    const VTKLegacyDataUnstructuredGrid* data;
};

/// Description class typedef
typedef core::factories::CallAutoDescription<VTKLegacyDataCallUnstructuredGrid>
    VTKLegacyDataCallUnstructuredGridDescription;

} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_VTKDATACALLUNSTRUCTUREDGRID_H_INCLUDED
