//
// VTILoader.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_VTILOADER_H_INCLUDED
#define MMPROTEINPLUGIN_VTILOADER_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/VTKImageData.h"
#include "vislib/Map.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"

#include <fstream>
#include <map>

typedef vislib::math::Vector<int, 2> Vec2i;
typedef vislib::math::Cuboid<float> Cubef;

// TODO handle 'compressor'

typedef unsigned int uint;

namespace megamol::protein {

/**
 * Class to load the *.vti data format used in the Visualization Toolkit.
 * File series have to be named after the following naming scheme
 *
 * prefix.%%%.vti,
 *
 * where the '%' stands for an arbitrary number of digits.
 * Files have to be named concurrently, since the module stops checking for
 * files if a concurrent filename is not present (e.g. if file.00.vti,
 * file.01.vti, and file.04.vti are in the input folder, the module stops
 * looking for files after file.01.vti).
 */
class VTILoader : public megamol::core::view::AnimDataModule {
public:
    /** Ctor */
    VTILoader();

    /** Dtor */
    ~VTILoader() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VTILoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Loader module for *.vti file format used by the Visualization \
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
     * Reads data from an ASCII buffer to a float array.
     *
     * @param buffIn  The input buffer containing the ascii data.
     * @param buffOut The output buffer containing the floats
     * @param sizeOut The size of the output buffer
     */
    void readDataAscii2Float(char* buffIn, float* buffOut, SIZE_T sizeOut);

    /**
     * Reads data from an binary buffer to a float array. Binary data has to be
     * encoded first since it uses base64 encoding.
     *
     * @param buffIn  The input buffer containing the ascii data.
     * @param buffOut The output buffer containing the floats
     * @param sizeOut The size of the output buffer
     */
    void readDataBinary2Float(char* buffIn, float* buffOut, SIZE_T sizeOut);

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

        /**
         * Answers a const pointer to the frame's data.
         *
         * @return A pointer to the data.
         */
        const protein_calls::VTKImageData* GetData() const {
            return &this->data;
        }

        /**
         * Answers the extent of the frame data.
         *
         * @return the extent of the frame data.
         */
        Cubeu GetWholeExtent() const {
            return this->data.GetWholeExtent();
        }

        /**
         * Answers the origin of the frame data.
         *
         * @return The origin of the frame data.
         */
        Vec3f GetOrigin() const {
            return this->data.GetOrigin();
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
            return this->data.PeekPointData(id, idx)->PeekData();
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

            return this->data.PeekPointData(arrayIdx, pieceIdx)->PeekData();
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

            return this->data.PeekCellData(id, idx)->PeekData();
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

            return this->data.PeekCellData(arrayIdx, pieceIdx)->PeekData();
        }

        /**
         * Answers the number of elements in the cell data array in the piece
         * 'pieceIdx' with the index 'dataIdx'.
         *
         * @param pieceIdx The piece's index
         * @param dataIdx  The data array's index
         * @return The number of elements in the data array.
         */
        size_t GetPieceCellArraySize(unsigned int pieceIdx, unsigned int dataIdx) {
            return this->data.GetPieceCellArraySize(dataIdx, pieceIdx);
        }

        /**
         * Answers the number of elements in the point data array in the piece
         * 'pieceIdx' with the index 'dataIdx'.
         *
         * @param pieceIdx The piece's index
         * @param dataIdx  The data array's index
         * @return The number of elements in the data array.
         */
        size_t GetPiecePointArraySize(unsigned int pieceIdx, unsigned int dataIdx) {
            return this->data.GetPiecePointArraySize(dataIdx, pieceIdx);
        }

        /**
         * Answers the spacing of the frame data.
         *
         * @return The spacing of the frame data.
         */
        Vec3f GetSpacing() const {
            return this->data.GetSpacing();
        }

        /**
         * Answers the extent of a piece.
         *
         * @return The extent of the requested piece.
         */
        vislib::math::Cuboid<unsigned int> GetPieceExtent(unsigned int pieceIdx) {
            return this->data.GetPieceExtent(pieceIdx);
        }

        /**
         * Set the frame Index.
         *
         * @param idx the index
         */
        void SetFrameIdx(uint idx) {
            this->frame = idx;
        }

        /**
         * Sets the extent of the frame data.
         *
         * @param wholeExtent The extent of the frame data.
         */
        inline void SetWholeExtent(vislib::math::Cuboid<uint> wholeExtent) {
            this->data.SetWholeExtent(wholeExtent);
        }

        /**
         * Sets the origin of the frame data.
         *
         * @param The origin of the frame data
         */
        inline void SetOrigin(Vec3f origin) {
            this->data.SetOrigin(origin);
        }

        /**
         * Sets the spacing of the frame data.
         */
        inline void SetSpacing(Vec3f spacing) {
            this->data.SetSpacing(spacing);
        }

        /**
         * Sets the number of pieces of the frame data.
         *
         * @param nPieces The number of pieces.
         */
        inline void SetNumberOfPieces(uint nPieces) {
            this->data.SetNumberOfPieces(nPieces);
        }

        /**
         * Sets the extent of a certain piece.
         *
         * @param pieceIdx The index of the piece.
         * @param extent The piece's extent.
         */
        inline void SetPieceExtent(unsigned int pieceIdx, Cubeu extent) {
            this->data.SetPieceExtent(pieceIdx, extent);
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const Frame& rhs);

        /**
         * Get point data array according to the array's id and the piece's
         * index. Might return NULL if the requested id is not in use.
         *
         * @param The arrays id.
         * @param The piece's index.
         * @return A pointer to the requested data array or NULL.
         */
        const protein_calls::VTKImageData::DataArray* PeekPointData(vislib::StringA id, unsigned int idx) {
            return this->data.PeekPointData(id, idx);
        }

        /**
         * Get cell data array according to the array's id and the piece's
         * index. Might return NULL if the requested id is not in use.
         *
         * @param The arrays id.
         * @param The piece's index.
         * @return A pointer to the requested data array or NULL.
         */
        const protein_calls::VTKImageData::DataArray* PeekCellData(vislib::StringA id, unsigned int idx) {
            return this->data.PeekCellData(id, idx);
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
        void SetPointData(const char* data, double min, double max, protein_calls::VTKImageData::DataArray::DataType t,
            vislib::StringA id, size_t nComponents, unsigned int pieceIdx) {
            this->data.SetPointData(data, min, max, t, id, nComponents, pieceIdx);
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
        void SetCellData(const char* data, double min, double max, protein_calls::VTKImageData::DataArray::DataType t,
            vislib::StringA id, size_t nComponents, unsigned int pieceIdx) {
            this->data.SetCellData(data, min, max, t, id, nComponents, pieceIdx);
        }

    private:
        protein_calls::VTKImageData data;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public protein_calls::VTIDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : protein_calls::VTIDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        /** Dtor. */
        ~Unlocker() override {
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
     * Convert a string to the respective integer value using string stream
     * objects.
     *
     * @param str The string
     * @return The integer
     */
    int string2int(vislib::StringA str);

    /**
     * Convert a string to the respective float value using string stream
     * objects.
     *
     * @param str The string
     * @return The float
     */
    float string2float(vislib::StringA str);


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


    /* The data set */

    /// The data hash
    SIZE_T hash;

    /// The byte order of the stored data (if binary)
    protein_calls::VTKImageData::ByteOrder byteOrder;

    /// The version (major, minor)
    Vec2i version;

    /// The data sets number of elements in each direction
    vislib::math::Cuboid<uint> wholeExtent;

    Vec3f origin;  ///> The data sets origin in world space coordinates
    Vec3f spacing; ///> The data sets spacing in each direction
    uint nPieces;  ///> The number of pieces in each frame

    vislib::StringA filenamesPrefix; ///> The prefix of the file series
    vislib::StringA filenamesSuffix; ///> The suffix og the file series
    uint filenamesDigits;            ///> The number of digits in the file names

    uint nFrames; ///> The number of frames in the data set
};


} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_VTILOADER_H_INCLUDED
