//
// VTKLegacyDataUnstructuredGrid.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#pragma once

#include "protein/AbstractVTKLegacyData.h"
#include "vislib/Array.h"

namespace megamol::protein {

/*
 * A class representing one frame of data given in the VTK legacy file format.
 */
class VTKLegacyDataUnstructuredGrid : public AbstractVTKLegacyData {

public:
    /** Enum describing possible cell types used by the VTK */
    enum CellType {
        VTK_VERTEX = 1,
        VTK_POLY_VERTEX,
        VTK_LINE,
        VTK_POLY_LINE,
        VTK_TRIANGLE,
        VTK_TRIANGLE_STRIP,
        VTK_POLYGON,
        VTK_PIXEL,
        VTK_QUAD,
        VTK_TETRA,
        VTK_VOXEL,
        VTK_HEXAHEDRON,
        VTK_WEDGE,
        VTK_PYRAMID
    };

    /** TODO */
    void AddPointData(const char* data, size_t nElements, size_t nComponents, DataType type, vislib::StringA name);

    /** CTor */
    VTKLegacyDataUnstructuredGrid();

    /** DTor */
    ~VTKLegacyDataUnstructuredGrid() override;

    /**
     * Answers the pointer to a point data attribute array specified by a name.
     *
     * @param idx The index of the seeked data array.
     * @return The pointer to the data, or NULL if the array is not there
     */
    const char* PeekCellDataByIndex(size_t idx) const;

    /**
     * Answers the pointer to a point data attribute array specified by a name.
     *
     * @param name The name of the seeked data array.
     * @return The pointer to the data, or NULL if the array is not there
     */
    const char* PeekCellDataByName(vislib::StringA name) const;

    /**
     * Answers the pointer to the cell index data
     *
     * @return The pointer to the data, or NULL if the array is not there
     */
    inline const int* PeekCells() const {
        //        // DEBUG Print vertex positions
        //        for (size_t p = 0; p < this->nPoints; ++p) {
        //            printf("!!%i: (%f %f %f)\n", p,
        //                    this->points[3*p+0],
        //                    this->points[3*p+1],
        //                    this->points[3*p+2]);
        //        }
        //        // END DEBUG
        return this->cells;
    }

    /**
     * Answers the pointer to the cell types
     *
     * @return The pointer to the data, or NULL if the array is not there
     */
    inline const CellType* PeekCellTypes() const {
        //        // DEBUG Print vertex positions
        //        for (size_t p = 0; p < this->nPoints; ++p) {
        //            printf("!!%i: (%f %f %f)\n", p,
        //                    this->points[3*p+0],
        //                    this->points[3*p+1],
        //                    this->points[3*p+2]);
        //        }
        //        // END DEBUG
        return this->cellTypes;
    }

    /**
     * Answers the pointer to the vertex positions
     *
     * @return The pointer to the data, or NULL if the array is not there
     */
    inline const float* PeekPoints() const {
        //        // DEBUG Print vertex positions
        //        for (size_t p = 0; p < this->nPoints; ++p) {
        //            printf("!!%i: (%f %f %f)\n", p,
        //                    this->points[3*p+0],
        //                    this->points[3*p+1],
        //                    this->points[3*p+2]);
        //        }
        //        // END DEBUG
        return this->points;
    }

    /**
     * Answers the pointer to a point data attribute array specified by a name.
     *
     * @param idx The index of the seeked data array.
     * @return The pointer to the data, or NULL if the array is not there
     */
    const AttributeArray* PeekPointDataByIndex(size_t idx) const;

    /**
     * Answers the pointer to a point data attribute array specified by a name.
     *
     * @param name The name of the seeked data array.
     * @return The pointer to the data, or NULL if the array is not there
     */
    const AttributeArray* PeekPointDataByName(vislib::StringA name) const;

    /**
     * Answers the number of vertices.
     *
     * @return The number of points
     */
    size_t GetNumberOfPoints() const {
        return this->nPoints;
    }

    /**
     * Answers the size of the cell index data
     *
     * @return The size of the cell index data
     */
    size_t GetCellDataSize() const {
        return this->nCellData;
    }

    /**
     * Answers the size of the cell index data
     *
     * @return The size of the cell index data
     */
    size_t GetNumberOfCells() const {
        return this->nCells;
    }

    /**
     * Copy vertex coordinates from a buffer.
     *
     * @param buff    The input buffer
     * @param nPoints The number of points
     */
    void SetPoints(const float* buff, size_t nPoints) {
        // (Re)allocate if necessary
        if (nPoints > this->nPoints) {
            if (this->points)
                delete[] this->points;
            this->points = new float[3 * nPoints];
        }
        this->nPoints = nPoints;
        memcpy(this->points, buff, this->nPoints * sizeof(float) * 3);
    }

    /**
     * Copy vertex coordinates from a buffer.
     *
     * @param buff The input buffer
     */
    void SetCellIndexData(const int* buff, size_t cellDataCnt) {
        // (Re)allocate if necessary
        if (cellDataCnt > this->nCellData) {
            if (this->cells)
                delete[] this->cells;
            this->cells = new int[cellDataCnt];
        }
        this->nCellData = cellDataCnt;
        memcpy(this->cells, buff, this->nCellData * sizeof(int));
    }

    /**
     * Copy vertex coordinates from a buffer.
     *
     * @param buff The input buffer
     */
    void SetCellTypes(const int* buff, size_t cellCnt) {
        // (Re)allocate if necessary
        if (cellCnt > this->nCells) {
            if (this->cellTypes)
                delete[] this->cellTypes;
            this->cellTypes = new CellType[cellCnt];
        }
        this->nCells = cellCnt;
        memcpy(this->cellTypes, buff, this->nCells * sizeof(CellType));
    }

    /** TODO */
    inline size_t GetPointDataCount() const {
        return this->pointData.Count();
    }

protected:
private:
    /// The array of points using the following format:
    /// x_0 y_0 z_0 x_1 y_1 z_1 ...
    float* points;

    /// The number of vertices
    size_t nPoints;

    /// The array of cells using the following format:
    /// numPoints_0 i_0 j_0 k_0 ....
    /// numPoints_1 i_1 j_1 k_1 ....
    /// ...
    int* cells;

    /// The number of cells
    size_t nCells;

    /// Array containing the different cell types for all cells
    CellType* cellTypes;

    /// The size of the cell index data
    size_t nCellData;

    /// Data attributes for the point data
    vislib::Array<AbstractVTKLegacyData::AttributeArray> pointData;

    /// Data attributes for the cell data
    vislib::Array<AbstractVTKLegacyData::AttributeArray> cellData;
};

} // namespace megamol::protein
