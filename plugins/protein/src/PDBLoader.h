/*
 * PDBLoader.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_PDBLOADER_H_INCLUDED
#define MMPROTEINPLUGIN_PDBLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MDDriverConnector.h"
#include "Stride.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/RunnableThread.h"
#include <filesystem>
#include <fstream>

#ifdef WITH_CURL
#include <curl/curl.h>
#endif

namespace megamol {
namespace protein {

/**
 * Data source for PDB files
 */

class PDBLoader : public megamol::core::view::AnimDataModule {
public:
    // AquariaLoader needs access to Callbacks
    friend class MultiPDBLoader;

    /** Ctor */
    PDBLoader();

    /** Dtor */
    ~PDBLoader() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "PDBLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers protein data.";
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
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Call callback to check whether data has been changed/needs update
     *
     * @param c The calling call
     *
     * @return whether data gas changed
     */
    bool dataChanged(core::Call& call) {
        return false; /*return solventResidues.IsDirty();*/
    }

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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
         * Encode a given int to a certain number of bits
         * TODO
         */
        void encodebits(char* outbuff, int bitsize, int bitoffset, unsigned int num);

        /**
         * Encode three integers (representing one coordinate).
         *
         * @param outbuff      buffer for the encoded integers
         * @param num_of_bits  the bitsize of the encoded integers
         * @param sizes        the ranges of value
         * @param inbuff       integers to be encoded
         * @param bitoffset    the bitoffset in the first byte
         */
        bool encodeints(char* outbuff, int num_of_bits, unsigned int sizes[], int inbuff[], unsigned int bitoffset);

        /**
         * Encode the frame and write it to the given XTC-file.
         *
         * @param outfile    The XTC-file.
         * @param precision  The precision of the encoded float coordinates.
         *
         * @return 'true' if the frame could be written
         */
        bool writeFrame(std::ofstream* outfile, float precision, float* minFloats, float* maxfloats);

        /**
         * Reads and decodes one frame of the data set from a given
         * xtc-file.
         *
         * @param file Pointer to the current frame in the xtc-file
         */
        void readFrame(std::fstream* file);

        /**
         * Calculates the number of bits needed to represent a given
         * integer value
         *
         * @param The integer value
         *
         * @return The number of bits
         */
        int sizeofint(int size);

        /**
         * Calculates the number of bits needed to represent 3 ints.
         *
         * @param sizes The range of the ints
         *
         * @return The needed number of bits
         */
        unsigned int sizeofints(unsigned int sizes[]);

        /**
         * Decodes integers from a given byte-array by calculating the
         * remainder and doing divisions with the maximum range.
         *
         * @param buff pointer to the byte buffer
         * @param offset the bit-offset within the first byte
         * @param num_of_bits the total number of bits to decode
         * @param sizes the range of the integers
         * @param nums array of the decoded integers
         */
        void decodeints(char* buff, int offset, int num_of_bits, unsigned int sizes[], int nums[]);

        /**
         * Interprets a given bit array as an integer.
         *
         * @param buff pointer to the byte buffer
         * @param offset the bit-offset within the first byte
         * @param bitsize the total number of bits
         *
         * @return the decoded integer
         */
        int decodebits(char* buff, int offset, int bitsize);

        /**
         * Reverse the order of bytes in a given char-array of 4 elements.
         *
         * @param num the char-array
         */
        void changeByteOrder(char* num);

        /**
         * Set the frame Index.
         *
         * @param idx the index
         */
        void setFrameIdx(int idx);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const Frame& rhs);

        /**
         * Set the atom count.
         *
         * @param atomCnt The atom count
         */
        inline void SetAtomCount(unsigned int atomCnt) {
            this->atomCount = atomCnt;
            this->atomPosition.SetCount(atomCnt * 3);
            this->bfactor.SetCount(atomCnt);
            this->charge.SetCount(atomCnt);
            this->occupancy.SetCount(atomCnt);
        }

        /**
         * Get the atom count.
         *
         * @return The atom count.
         */
        inline unsigned int AtomCount() const {
            return this->atomCount;
        }

        /**
         * Assign a position to the array of positions.
         */
        bool SetAtomPosition(unsigned int idx, float x, float y, float z);

        /**
         * Assign a bfactor to the array of bfactors.
         */
        bool SetAtomBFactor(unsigned int idx, float val);

        /**
         * Assign a charge to the array of charges.
         */
        bool SetAtomCharge(unsigned int idx, float val);

        /**
         * Assign a occupancy to the array of occupancies.
         */
        bool SetAtomOccupancy(unsigned int idx, float val);

        /**
         * Set the b-factor range.
         *
         * @param min    The minimum b-factor.
         * @param max    The maximum b-factor.
         */
        void SetBFactorRange(float min, float max) {
            this->minBFactor = min;
            this->maxBFactor = max;
        }

        /**
         * Set the minimum b-factor.
         *
         * @param min    The minimum b-factor.
         */
        void SetMinBFactor(float min) {
            this->minBFactor = min;
        }

        /**
         * Set the maximum b-factor.
         *
         * @param max    The maximum b-factor.
         */
        void SetMaxBFactor(float max) {
            this->maxBFactor = max;
        }

        /**
         * Set the charge range.
         *
         * @param min    The minimum charge.
         * @param max    The maximum charge.
         */
        void SetChargeRange(float min, float max) {
            this->minCharge = min;
            this->maxCharge = max;
        }

        /**
         * Set the minimum charge.
         *
         * @param min    The minimum charge.
         */
        void SetMinCharge(float min) {
            this->minCharge = min;
        }

        /**
         * Set the maximum charge.
         *
         * @param max    The maximum charge.
         */
        void SetMaxCharge(float max) {
            this->maxCharge = max;
        }

        /**
         * Set the occupancy range.
         *
         * @param min    The minimum occupancy.
         * @param max    The maximum occupancy.
         */
        void SetOccupancyRange(float min, float max) {
            this->minOccupancy = min;
            this->maxOccupancy = max;
        }

        /**
         * Set the minimum occupancy.
         *
         * @param min    The minimum occupancy.
         */
        void SetMinOccupancy(float min) {
            this->minOccupancy = min;
        }

        /**
         * Set the maximum occupancy.
         *
         * @param max    The maximum occupancy.
         */
        void SetMaxOccupancy(float max) {
            this->maxOccupancy = max;
        }

        /**
         * Get a reference to the array of atom positions.
         *
         * @return The atom position array.
         */
        const float* AtomPositions() {
            return this->atomPosition.PeekElements();
        }

        /**
         * Get a reference to the array of atom b-factors.
         *
         * @return The atom b-factor array.
         */
        float* AtomBFactor() {
            return &this->bfactor[0];
        }

        /**
         * Get a reference to the array of atom charges.
         *
         * @return The atom charge array.
         */
        const float* AtomCharge() {
            return this->charge.PeekElements();
        }

        /**
         * Get a reference to the array of atom occupancies.
         *
         * @return The atom occupancy array.
         */
        const float* AtomOccupancy() {
            return this->occupancy.PeekElements();
        }

        /**
         * Get the maximum b-factor of this frame.
         *
         * @return The maximum b-factor.
         */
        float MaxBFactor() const {
            return this->maxBFactor;
        }

        /**
         * Get the minimum b-factor of this frame.
         *
         * @return The minimum b-factor.
         */
        float MinBFactor() const {
            return this->minBFactor;
        }

        /**
         * Get the maximum b-factor of this frame.
         *
         * @return The maximum b-factor.
         */
        float MaxCharge() const {
            return this->maxCharge;
        }

        /**
         * Get the minimum charge of this frame.
         *
         * @return The minimum charge.
         */
        float MinCharge() const {
            return this->minCharge;
        }

        /**
         * Get the maximum occupancy of this frame.
         *
         * @return The maximum occupancy.
         */
        float MaxOccupancy() const {
            return this->maxOccupancy;
        }

        /**
         * Get the minimum occupancy of this frame.
         *
         * @return The minimum occupancy.
         */
        float MinOccupancy() const {
            return this->minOccupancy;
        }

    private:
        /** The atom count */
        unsigned int atomCount;

        /** The atom positions */
        vislib::Array<float> atomPosition;

        /** The atom b-factors */
        vislib::Array<float> bfactor;

        /** The atom charges */
        vislib::Array<float> charge;

        /** The atom occupancy */
        vislib::Array<float> occupancy;

        /** The maximum b-factor */
        float maxBFactor;
        /** The minimum b-factor */
        float minBFactor;

        /** The maximum carge */
        float maxCharge;
        /** The minimum charge */
        float minCharge;

        /** The maximum occupancy */
        float maxOccupancy;
        /** The minimum occupancy */
        float minOccupancy;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public megamol::protein_calls::MolecularDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : megamol::protein_calls::MolecularDataCall::Unlocker(), frame(&frame) {
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

#ifdef WITH_CURL
    /**
     *
     * @param filename for the pdb file in the database
     */
    std::string loadFromPDB(std::string filename);

    /**
     *
     *
     */
    //size_t WriteMemoryCallback(char* buf, size_t size, size_t nmemb, void* up);

#endif

    /**
     * Loads a PDB file.
     *
     * @param filename The path to the file to load.
     */
    void loadFile(const std::filesystem::path& filename);

    /**
     * Loads a file containing information about the cap(s).
     *
     * @param filename The path to the file to load.
     */
    void loadFileCap(const std::filesystem::path& filename);

    /**
     * Parse one atom entry.
     *
     * @param atomEntry The atom entry string.
     * @param atom      The number of the current atom.
     * @param frame     The number of the current frame.
     */
    void parseAtomEntry(vislib::StringA& atomEntry, unsigned int atom, unsigned int frame,
        vislib::Array<vislib::TString>& solventResidueNames);

    /**
     * Parse the CRYST entry in a PDB file
     *
     * @param bboxEntry
     */
    void parseBBoxEntry(vislib::StringA& bboxEntry);

    /**
     * Get the radius of the element.
     *
     * @param name The name of the atom type.
     * @return The radius of the element in Angstrom.
     */
    float getElementRadius(vislib::StringA name);

    /**
     * Get the color of the element.
     *
     * @param name The name of the atom type.
     * @return The color of the element.
     */
    vislib::math::Vector<unsigned char, 3> getElementColor(vislib::StringA name);

    /**
     * Parse one atom entry and set the position of the current atom entry
     * to the frame.
     *
     * @param atomEntry The atom entry string.
     * @param atom      The number of the current atom.
     * @param frame     The number of the current frame.
     */
    void setAtomPositionToFrame(vislib::StringA& atomEntry, unsigned int atom, unsigned int frame);

    /**
     * Search for connections in the given residue and add them to the
     * global connection array.
     *
     * @param resIdx The index of the residue.
     * @param resIdx The index of the reference frame.
     */
    void MakeResidueConnections(unsigned int resIdx, unsigned int frame);

    /**
     * Search for connections between two residues.
     *
     * @param resIdx0   The index of the first residue.
     * @param resIdx1   The index of the second residue.
     * @param resIdx    The index of the reference frame.
     *
     * @return 'true' if connections were found, 'false' otherwise.
     */
    bool MakeResidueConnections(unsigned int resIdx0, unsigned int resIdx1, unsigned int frame);

    /**
     * Check if the residue is an amino acid.
     *
     * @return 'true' if resName specifies an amino acid, 'false' otherwise.
     */
    bool IsAminoAcid(vislib::StringA resName);
    /**
     * Reset all data containers.
     */
    void resetAllData();

    /**
     * Read the number of frames from the XTC file
     *
     * @return 'true' if the file could be loaded, otherwise 'false'
     */
    bool readNumXTCFrames();

    /**
     * Writes the frames of the current PDB-file (beginning with second
     * frame) into a new compressed XTC-file.
     *
     * The PDB-file has to be fully loaded before because the data-sets
     * bounding box is needed.
     *
     * @param filename The name of the output file.
     */
    void writeToXtcFile(const vislib::TString& filename);


    // -------------------- variables --------------------

    /** The pdb file name slot */
    core::param::ParamSlot pdbFilenameSlot;
    /** The xtc file name slot */
    core::param::ParamSlot xtcFilenameSlot;
    /** The cap file name slot */
    core::param::ParamSlot capFilenameSlot;
    /** The data callee slot */
    core::CalleeSlot dataOutSlot;
    /** caller slot */
    core::CallerSlot forceDataCallerSlot;

    /** The maximum frame slot */
    core::param::ParamSlot maxFramesSlot;
    /** The STRIDE usage flag slot */
    core::param::ParamSlot strideFlagSlot;
    /** slot to specify a ;-list of residues to be merged into separate chains ... */
    core::param::ParamSlot solventResidues;
    /** Determine whether to use the PDB bbox */
    core::param::ParamSlot calcBBoxPerFrameSlot;
    /** Determine whether to use the PDB bbox */
    core::param::ParamSlot calcBondsSlot;
    /** Determine whether to recompute STRIDE each frame */
    core::param::ParamSlot recomputeStridePerFrameSlot;

    /** The data */
    vislib::Array<Frame*> data;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;
    vislib::Array<vislib::math::Cuboid<float>> bboxPerFrame;

    /** The data hash */
    SIZE_T datahash;

    /** Stores for each atom the index of its type */
    vislib::Array<unsigned int> atomTypeIdx;

    /** Stores for each atom its former index from the pdb file */
    vislib::Array<int> atomFormerIdx;

    /* Residue index per atom - may be undefined (-1) */
    vislib::Array<int> atomResidueIdx;

    /** The array of atom types */
    vislib::Array<megamol::protein_calls::MolecularDataCall::AtomType> atomType;

    /** The array of residues */
    vislib::Array<megamol::protein_calls::MolecularDataCall::Residue*> residue;

    /** The array of residue type names */
    vislib::Array<vislib::StringA> residueTypeName;

    /** residue indices marked as solvent */
    vislib::Array<unsigned int> solventResidueIdx;

    /** The array of molecules */
    vislib::Array<megamol::protein_calls::MolecularDataCall::Molecule> molecule;

    /** The array of chains */
    vislib::Array<megamol::protein_calls::MolecularDataCall::Chain> chain;

    /** The array stores the begining end ending of a cap. */
    vislib::Array<std::pair<int, int>> cap_chain;

    /**
     * Stores the connectivity information (i.e. subsequent pairs of atom
     * indices)
     */
    vislib::Array<unsigned int> connectivity;

    /** Stores the current residue sequence number while loading */
    unsigned int resSeq;

    /** Stores the current molecule count while loading */
    unsigned int molIdx;

    /** Stride secondary structure computation */
    Stride* stride;
    /** Flag whether secondary structure is available */
    bool secStructAvailable;

    // Temporary variables for molecular chains
    vislib::Array<unsigned int> chainFirstRes;
    vislib::Array<unsigned int> chainResCount;
    vislib::Array<char> chainName;
    vislib::Array<megamol::protein_calls::MolecularDataCall::Chain::ChainType> chainType;
    char chainId;

    /** the number of frames */
    unsigned int numXTCFrames;
    /** the byte offset of all frames */
    vislib::Array<unsigned int> XTCFrameOffset;
    /** Flag whether the current xtc-filename is valid */
    bool xtcFileValid;

    /** MDDriverLoader object for connecting to MDDriver */
    vislib::sys::RunnableThread<MDDriverConnector>* mdd;

    /** Per atom filter information to be used by MolecularDataCall */
    vislib::Array<int> atomVisibility;

    /** Storage of the pdb filename */
    std::filesystem::path pdbfilename;
};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_PDBLOADER_H_INCLUDED
