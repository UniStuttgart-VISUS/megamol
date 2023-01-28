//
// VTIDataCall.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 16, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCALLPLUGIN_VTIDATACALL_H_INCLUDED
#define MMPROTEINCALLPLUGIN_VTIDATACALL_H_INCLUDED
#pragma once

#include "Interpol.h"
#include "VTKImageData.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
typedef vislib::math::Vector<int, 3> Vec3i;

namespace megamol::protein_calls {

class VTIDataCall : public core::AbstractGetData3DCall {

public:
    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /** Ctor. */
    VTIDataCall();

    /** Dtor. */
    ~VTIDataCall() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VTIDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call for VTK image data.";
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
    void SetData(const VTKImageData* data) {
        this->data = data;
    }

    /**
     * Answers the extent of the data set.
     *
     * @return The extent of the data set.
     */
    inline vislib::math::Cuboid<uint> GetWholeExtent() const {
        return this->data->GetWholeExtent();
    }

    /**
     * Answers the origin of the lattice.
     *
     * @return The origin of the lattice.
     */
    inline Vec3f GetOrigin() const {
        return this->data->GetOrigin();
    }

    /**
     * Answers the spacing of the lattice.
     *
     * @return The spacing of the lattice.
     */
    inline Vec3f GetSpacing() const {
        return this->data->GetSpacing();
    }

    /**
     * Answers the grid size.
     *
     * @return The gridsize.
     */
    inline Vec3i GetGridsize() const {
        return Vec3i(this->data->GetWholeExtent().Width() + 1, this->data->GetWholeExtent().Height() + 1,
            this->data->GetWholeExtent().Depth() + 1);
    }

    /**
     * Answers the extent of a piece.
     *
     * @pieceIdx The pieces index
     * @return The extent of the requested piece.
     */
    Cubeu GetPieceExtent(unsigned int pieceIdx) const {
        return this->data->GetPieceExtent(pieceIdx);
    }

    /**
     * Get point data array according to the array's id and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetPointDataByName(vislib::StringA id, unsigned int idx) const {
        return this->data->PeekPointData(id, idx)->PeekData();
    }

    /**
     * Get point data array according to the array's index and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetPointDataByIdx(unsigned int arrayIdx, unsigned int pieceIdx) const {

        return this->data->PeekPointData(arrayIdx, pieceIdx)->PeekData();
    }

    /**
     * Get cell data array according to the array's id and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetCellDataByName(vislib::StringA id, unsigned int idx) {

        return this->data->PeekCellData(id, idx)->PeekData();
    }

    /**
     * Get cell data array according to the array's index and the piece's
     * index. Might return NULL if the requested id is not in use.
     *
     * @param The arrays id.
     * @param The piece's index.
     * @return A pointer to the requested data array or NULL.
     */
    const char* GetCellDataByIdx(unsigned int arrayIdx, unsigned int pieceIdx) const {

        return this->data->PeekCellData(arrayIdx, pieceIdx)->PeekData();
    }

    /**
     * Answers the number of pieces.
     *
     * @return The number of pieces.
     */
    inline uint GetNumberOfPieces() {
        return this->data->GetNumberOfPieces();
    }

    /**
     * Answers the number of data arrays present in the point data of the piece
     * with the index 'pieceIdx'.
     *
     * @param pieceIdx The index of the piece.
     * @return The number of data arrays in the specified piece.
     */
    inline size_t GetArrayCntOfPiecePointData(unsigned int pieceIdx) const {
        return this->data->GetArrayCntOfPiecePointData(pieceIdx);
    }

    /**
     * Answers the number of data arrays present in the cell data present in the
     * piece with the index 'pieceIdx'.
     *
     * @param pieceIdx The index of the piece.
     * @return The number of data arrays in the specified piece.
     */
    inline size_t GetArrayCntOfPieceCellData(unsigned int pieceIdx) const {
        return this->data->GetArrayCntOfPieceCellData(pieceIdx);
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
        return this->data->GetPointDataArrayNumberOfComponents(dataIdx, pieceIdx);
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
        return this->data->GetCellDataArrayNumberOfComponents(dataIdx, pieceIdx);
    }

    /**
     * Answers the name of the cell data array with the index 'dataIdx' of the
     * piece with the piece index 'pieceIdx'.
     *
     * @return The name of the cell data array.
     */
    inline vislib::StringA GetCellDataArrayId(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetCellDataArrayId(dataIdx, pieceIdx);
    }

    /**
     * Answers the name of the point data array with the index 'dataIdx' of the
     * piece with the piece index 'pieceIdx'.
     *
     * @return The name of the point data array.
     */
    inline vislib::StringA GetPointDataArrayId(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPointDataArrayId(dataIdx, pieceIdx);
    }

    /**
     * Answers the data type of the data array with the index 'dataIdx' of
     * the point data of the piece 'pieceIdx'.
     *
     * @return The type of the data array.
     */
    inline VTKImageData::DataArray::DataType GetPiecePointArrayType(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPiecePointArrayType(dataIdx, pieceIdx);
    }

    /**
     * Answers the data type of the data array with the index 'dataIdx' of
     * the cell data of the piece 'pieceIdx'.
     *
     * @return The type of the data array.
     */
    inline VTKImageData::DataArray::DataType GetPieceCellArrayType(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPieceCellArrayType(dataIdx, pieceIdx);
    }

    /**
     * Answers the minimum point data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The minimum value.
     */
    double GetPointDataArrayMin(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPointDataArrayMin(dataIdx, pieceIdx);
    }

    /**
     * Answers the maximum point data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The maximum value.
     */
    double GetPointDataArrayMax(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPointDataArrayMax(dataIdx, pieceIdx);
    }

    /**
     * Answers the minimum cell data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The minimum value.
     */
    double GetCellDataArrayMin(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetCellDataArrayMin(dataIdx, pieceIdx);
    }

    /**
     * Answers the maximum cell data value of the specified data array.
     *
     * @param The index of the data array.
     * @param The index of the piece.
     * @return The maximum value.
     */
    double GetCellDataArrayMax(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetCellDataArrayMax(dataIdx, pieceIdx);
    }

    /**
     * Answers the number of elements in a specified point data array.
     *
     * @param dataIdx The index of the specified array.
     * @param pieceIdx The piece's index.
     * @return The number of elements in the array.
     */
    inline size_t GetPiecePointArraySize(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPiecePointArraySize(dataIdx, pieceIdx);
    }

    /**
     * Answers the number of elements in a specified cell data array.
     *
     * @param dataIdx The index of the specified array.
     * @param pieceIdx The piece's index.
     * @return The number of elements in the array.
     */
    inline size_t GetPieceCellArraySize(unsigned int dataIdx, unsigned int pieceIdx) const {
        return this->data->GetPieceCellArraySize(dataIdx, pieceIdx);
    }

private:
    float calltime; ///> The exact requested/stored call time
    //size_t frameCnt;           ///> The number of frames
    const VTKImageData* data; ///> The current frame's image data (pointer)
};

/// Description class typedef
typedef core::factories::CallAutoDescription<VTIDataCall> VTIDataCallDescription;

} // namespace megamol::protein_calls

#endif // MMPROTEINCALLPLUGIN_VTIDATACALL_H_INCLUDED
