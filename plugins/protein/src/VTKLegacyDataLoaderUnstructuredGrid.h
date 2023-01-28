//
// VTKLegacyDataLoaderUnstructuredGrid.h
//
// Copyright (C) 2013-2018 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINPLUGIN_VTKLEGACYDATALOADERUNSTRUCTUREDGRID_H_INCLUDED
#define MMPROTEINPLUGIN_VTKLEGACYDATALOADERUNSTRUCTUREDGRID_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "mmstd/data/AnimDataModule.h"
#include "protein/VTKLegacyDataCallUnstructuredGrid.h"
#include "protein/VTKLegacyDataUnstructuredGrid.h"
#include "vislib/math/Cuboid.h"
typedef vislib::math::Cuboid<float> Cubef;

namespace megamol::protein {

/*
 * A class for loading VTK legacy unstructured grid data from *.vtk files
 */
class VTKLegacyDataLoaderUnstructuredGrid : public core::view::AnimDataModule {

public:
    /** CTor */
    VTKLegacyDataLoaderUnstructuredGrid();

    /** DTor */
    ~VTKLegacyDataLoaderUnstructuredGrid() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VTKLegacyDataLoaderUnstructuredGrid";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Loader module for *.vtk file format used by the Visualization \
                Toolkit.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
     */
    void release() override;

    /**
     * Call callback to get the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Loads a *.vti file.
     *
     * @param filename The files name.
     * @return 'True' on success, 'false', otherwise
     */
    bool loadFile(const vislib::StringA& filename);

    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    Frame* constructFrame() const override;

    /**
     * Loads one frame of the data set into the given 'frame' object. This
     * method may be invoked from another thread. You must take
     * precausions in case you need synchronised access to shared
     * ressources.
     *
     * @param frame The frame to be loaded.
     * @param idx The index of the frame to be loaded.
     */
    void loadFrame(Frame* frame, unsigned int idx) override;

private:
    /**
     * Storage of frame data
     */
    class Frame : public megamol::core::view::AnimDataModule::Frame {
    public:
        /** Ctor */
        Frame(megamol::core::view::AnimDataModule& owner);

        /** Dtor */
        ~Frame() override;

        void AddPointData(const char* data, size_t nElements, size_t nComponents, AbstractVTKLegacyData::DataType type,
            vislib::StringA name) {

            // Add new element to point data array
            this->data.AddPointData(data, nElements, nComponents, type, name);
        }

        /**
         * Answers a const pointer to the frame's data.
         *
         * @return A pointer to the data.
         */
        const VTKLegacyDataUnstructuredGrid* GetData() const {
            return &this->data;
        }

        /**
         * Get cell data array according to the array's id and the piece's
         * index. Might return NULL if the requested id is not in use.
         *
         * @param The arrays id.
         * @return A pointer to the requested data array or NULL.
         */
        const char* GetCellDataByName(vislib::StringA id) const {
            return this->data.PeekCellDataByName(id);
        }

        /**
         * Answers the encoding used in this data.
         *
         * @return The encoding of the data
         */
        AbstractVTKLegacyData::DataEncoding GetEncoding() const {
            return this->data.GetEncoding();
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
            return this->data.PeekCellDataByIndex(arrayIdx);
        }

        /**
         * Answers the number of vertices.
         *
         * @return The number of points
         */
        size_t GetNumberOfPoints() const {
            return this->data.GetNumberOfPoints();
        }

        /**
         * Sets the data encoding (either ascii or binary).
         *
         * @param encoding The data encoding
         */
        void SetEncoding(AbstractVTKLegacyData::DataEncoding encoding) {
            this->data.SetDataEncoding(encoding);
        }

        /**
         * Copy vertex coordinates from a buffer.
         *
         * @param buff The input buffer
         */
        void SetCellIndexData(const int* buff, size_t cellDataCnt) {
            this->data.SetCellIndexData(buff, cellDataCnt);
        }

        /**
         * Copy vertex coordinates from a buffer.
         *
         * @param buff The input buffer
         */
        void SetCellTypes(const int* buff, size_t cellCnt) {
            this->data.SetCellTypes(buff, cellCnt);
        }

        /**
         * Sets the geometry/topology information.
         *
         * @param topology The data's topology.
         */
        void SetTopology(AbstractVTKLegacyData::DataGeometry topology) {
            this->data.SetDataGeometry(topology);
        }

        /**
         * Set the frame Index.
         *
         * @param idx the index
         */
        void SetFrameIdx(unsigned int idx) {
            this->frame = idx;
        }

        /**
         * Copy vertex coordinates from a buffer.
         *
         * @param buff    The input buffer
         * @param nPoints The number of points
         */
        void SetPoints(const float* buff, size_t nPoints) {
            this->data.SetPoints(buff, nPoints);
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const Frame& rhs);

        /** TODO */
        inline size_t GetPointDataCount() const {
            return this->data.GetPointDataCount();
        }

        /**
         * Answers the pointer to a point data attribute array specified by a name.
         *
         * @param idx The index of the seeked data array.
         * @return The pointer to the data, or NULL if the array is not there
         */
        const AbstractVTKLegacyData::AttributeArray* PeekPointDataByIndex(size_t idx) const {
            return this->data.PeekPointDataByIndex(idx);
        }

        /**
         * Answers the pointer to a point data attribute array specified by a name.
         *
         * @param name The name of the seeked data array.
         * @return The pointer to the data, or NULL if the array is not there
         */
        const AbstractVTKLegacyData::AttributeArray* PeekPointDataByName(vislib::StringA name) const {
            return this->data.PeekPointDataByName(name);
        }

    private:
        /// The data object containing the VTK unstructured grid data
        VTKLegacyDataUnstructuredGrid data;
    };

    /**
     * Helper class to unlock frame data when
     * 'VTKLegacyDataCallUnstructuredGrid' is used.
     */
    class VTKUnlocker : public megamol::core::AbstractGetData3DCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        VTKUnlocker(Frame& frame) : megamol::core::AbstractGetData3DCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        /** Dtor. */
        ~VTKUnlocker() override {
            this->Unlock();
            ASSERT(this->frame == NULL);
        }

        /** Unlocks the data */
        void Unlock() override {
            if (this->frame != NULL) {
                this->frame->Unlock();
                this->frame = NULL; // DO NOT DELETE!
            }
        }

    private:
        /** The frame to unlock */
        Frame* frame;
    };

    /**
     * Answers whether the given char is a whitespace character.
     *
     * @param c The char to be tested
     * @return 'True' if c is a white space character.
     */
    bool isWhiteSpaceChar(char c) {
        switch (c) {
        case ' ':
            return true; // TODO extend
        case '\n':
            return true;
        default:
            return false;
        }
    }

    /**
     * Answers whether the given char is a newline character.
     *
     * @param c The char to be tested
     * @return 'True' if c is a white space character.
     */
    bool isNewlineChar(char c) {
        switch (c) {
        case '\n':
            return true; // TODO extend
        default:
            return false;
        }
    }

    /** TODO */
    void seekNextLine(char*& buffPt) {
        // Seek the end of the current line
        while (!this->isNewlineChar(*buffPt)) {
            buffPt++;
        }
        buffPt++;
    }

    /** TODO */
    void readASCIIFloats(char*& buffPt, float* out, size_t cnt);

    /** TODO */
    void readASCIIInts(char*& buffPt, int* out, size_t cnt);

    /** TODO */
    vislib::StringA readCurrentLine(char* buffPt) {
        size_t len1 = 0;
        // Seek the end of the current line
        while (!this->isNewlineChar(buffPt[len1])) {
            //            printf("%c\n", buffPt[len1]);
            len1++;
        }
        vislib::StringA line(buffPt, (int)len1);
        return line;
    }

    /** TODO */
    void seekNextToken(char*& buffPt) {
        // Seek the end of the current token
        while (!this->isWhiteSpaceChar(*buffPt)) {
            buffPt++;
        }
        // Seek the beginning of the next token
        while (this->isWhiteSpaceChar(*buffPt)) {
            buffPt++;
        }
    }

    /** TODO */
    vislib::StringA readNextToken(char*& buffPt) {
        size_t len = 0;
        // Skip non-white-space-characters
        while (!this->isWhiteSpaceChar(buffPt[len])) {
            len++;
        }
        //        printf("TOKENSTR %s\n", vislib::StringA(buffPt, len).PeekBuffer());
        return vislib::StringA(buffPt, (int)len);
    }

    /** TODO */
    void readCells(char*& buffPt, core::view::AnimDataModule::Frame* frame);

    /** TODO */
    void readCellTypes(char*& buffPt, core::view::AnimDataModule::Frame* frame);

    /** TODO */
    void readHeaderData(char*& buffPt, core::view::AnimDataModule::Frame* frame);

    /** TODO */
    void readPoints(char*& buffPt, core::view::AnimDataModule::Frame* frame);

    /** TODO */
    void readFieldData(char*& buffPt, core::view::AnimDataModule::Frame* frame, size_t numArrays);

    /** TODO */
    void readDataArray(char*& buffPt, core::view::AnimDataModule::Frame* frame, size_t nTupels, size_t nComponents,
        AbstractVTKLegacyData::DataAssociation);

    /** TODO */
    void swapBytes(char* buffPt, size_t stride, size_t cnt);


    /// The data callee slot
    core::CalleeSlot dataOutSlot;


    /* Parameter slots */

    /// Parameter slot containing path to the data file
    core::param::ParamSlot filenameSlot;

    /// Parameter slot indicatin the maximum number of frames to be loaded
    core::param::ParamSlot maxFramesSlot;

    /// Parameter slot indicatin the first frame to be loaded
    core::param::ParamSlot frameStartSlot;

    /// Parameter slot for the maximum cache size
    core::param::ParamSlot maxCacheSizeSlot;

    /// Parameter slot for the name of the data set to be sent with MultiparticleDataCall
    core::param::ParamSlot mpdcAttributeSlot;

    /// Parameter slot for the global radius of all particles to be sent with MultiparticleDataCall
    core::param::ParamSlot globalRadiusParam;

    /* The data set */

    /// The data hash
    SIZE_T hash;

    vislib::StringA filenamesPrefix; ///> The prefix of the file series
    vislib::StringA filenamesSuffix; ///> The suffix og the file series
    size_t filenamesDigits;          ///> The number of digits in the file names

    /// The number of frames
    size_t nFrames;

    /// Flag that determines whether point data is being read
    bool readPointData;

    /// Flag that determines whether cell data is being read
    bool readCellData;

    /// The world space bounding box
    Cubef bbox;
};

} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_VTKLEGACYDATALOADERUNSTRUCTUREDGRID_H_INCLUDED
