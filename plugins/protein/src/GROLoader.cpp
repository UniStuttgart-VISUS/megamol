/*
 * GROLoader.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "GROLoader.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "mmcore/utility/sys/MemmappedFile.h"
#include "stdafx.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/SmartPtr.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/types.h"
#include <ctime>
#include <fstream>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

#define SOLVENT_CHAIN_IDENTIFIER 127

/*
 * GROLoader::Frame::Frame
 */
GROLoader::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner)
        , atomCount(0)
        , maxBFactor(0)
        , minBFactor(0)
        , maxCharge(0)
        , minCharge(0)
        , maxOccupancy(0)
        , minOccupancy(0) {
    // Intentionally empty
}

/*
 * GROLoader::Frame::~Frame
 */
GROLoader::Frame::~Frame(void) {}

/*
 * GROLoader::Frame::operator==
 */
bool GROLoader::Frame::operator==(const GROLoader::Frame& rhs) {
    // TODO: extend this accordingly
    return true;
}

/*
 * interpret a given bit array as an integer
 */
int GROLoader::Frame::decodebits(char* buff, int offset, int bitsize) {

    int num = 0;
    int mask = (1 << bitsize) - 1; // '1'^bitsize

    // change byte order
    char tmpBuff[] = {buff[3], buff[2], buff[1], buff[0]};

    // interprete char-array as an integer
    num = *(int*)tmpBuff;
    // cut off right offset
    num = num >> (32 - (offset + bitsize));

    // cut off left offset by using only 'bitsize' first bits
    num = num & mask;

    return num;
}

/*
 * decodeints
 */
void GROLoader::Frame::decodeints(char* buff, int offset, int num_of_bits, unsigned int sizes[], int nums[]) {


    int bytes[32];
    int i, j, num_of_bytes, p, num;

    bytes[1] = bytes[2] = bytes[3] = 0;
    num_of_bytes = 0;

    while (num_of_bits > 8) {
        // note: bit-offset stays the same
        bytes[num_of_bytes] = decodebits(buff, offset, 8);
        buff++;
        num_of_bytes++;
        num_of_bits -= 8;
    }
    if (num_of_bits > 0) {
        bytes[num_of_bytes++] = decodebits(buff, offset, num_of_bits);
    }

    // get num[2] and num[1]
    for (i = 2; i > 0; i--) {

        num = 0;
        for (j = num_of_bytes - 1; j >= 0; j--) {
            num = (num << 8) | bytes[j];
            p = num / sizes[i];
            bytes[j] = p;
            num = num - p * sizes[i];
        }
        nums[i] = num;
    }

    // get num[0]
    nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

/*
 * sizeofints
 */
unsigned int GROLoader::Frame::sizeofints(unsigned int sizes[]) {

    int i, num;
    unsigned int num_of_bytes, num_of_bits;
    unsigned int bytes[32], bytecnt, tmp;

    num_of_bytes = 1;
    bytes[0] = 1;
    num_of_bits = 0;

    for (i = 0; i < 3; i++) {
        tmp = 0;
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while (tmp != 0) {
            bytes[bytecnt++] = tmp & 0xff;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }

    num = 1;
    num_of_bytes--;
    while (bytes[num_of_bytes] >= (unsigned int)num) {
        num_of_bits++;
        num *= 2;
    }

    return num_of_bits + num_of_bytes * 8;
}

/*
 * sizeofint
 */
int GROLoader::Frame::sizeofint(int size) {
    unsigned int num = 1;
    int num_of_bits = 0;

    while ((unsigned int)size >= num && num_of_bits < 32) {
        num_of_bits++;
        num <<= 1;
    }
    return num_of_bits;
}

/*
 * change byte-order
 */
void GROLoader::Frame::changeByteOrder(char* num) {

    char temp;
    temp = num[0];
    num[0] = num[3];
    num[3] = temp;
    temp = num[1];
    num[1] = num[2];
    num[2] = temp;
}

/*
 * set the frames index
 */
void GROLoader::Frame::setFrameIdx(int idx) {
    this->frame = idx;
}

/*
 * encode the frame and write it to outfile
 */
// TODO: handle the usage of large numbers
// TODO: no compression for three atoms or less
bool GROLoader::Frame::writeFrame(std::ofstream* outfile, float precision, float* minFloats, float* maxFloats) {

    unsigned int i;
    int minInt[3], maxInt[3];
    unsigned sizes[3];
    int thiscoord[3];
    unsigned int bitsize;
    unsigned int byteSize;

    // write date to outfile
    int date = 1995;
    changeByteOrder((char*)&date);
    outfile->write((char*)&date, 4);

    // write number of atoms to outfile
    int atomCount = AtomCount();
    changeByteOrder((char*)&atomCount);
    outfile->write((char*)&atomCount, 4);

    // write simulation step to outfile
    int step = frame;
    changeByteOrder((char*)&step);
    outfile->write((char*)&step, 4);

    // write simulation time to outfile
    float simtime = (float)frame;
    changeByteOrder((char*)&simtime);
    outfile->write((char*)&simtime, 4);

    precision /= 10.0;

    // get the range of values
    minInt[0] = (int)(minFloats[0] * precision + 1);
    minInt[1] = (int)(minFloats[1] * precision + 1);
    minInt[2] = (int)(minFloats[2] * precision + 1);
    maxInt[0] = (int)(maxFloats[0] * precision + 1);
    maxInt[1] = (int)(maxFloats[1] * precision + 1);
    maxInt[2] = (int)(maxFloats[2] * precision + 1);

    sizes[0] = maxInt[0] - minInt[0] + 1;
    sizes[1] = maxInt[1] - minInt[1] + 1;
    sizes[2] = maxInt[2] - minInt[2] + 1;

    // calculate the bitsize of one coordinate within this range
    bitsize = sizeofints(sizes);

    // write the bounding box to outfile
    float* box = new float[9];
    box[0] = (float)sizes[0] / precision;
    box[1] = 0.0;
    box[2] = 0.0;
    box[3] = 0.0;
    box[4] = (float)sizes[1] / precision;
    box[5] = 0.0;
    box[6] = 0.0;
    box[7] = 0.0;
    box[8] = (float)sizes[2] / precision;

    for (i = 0; i < 9; i++)
        changeByteOrder((char*)&box[i]);

    outfile->write((char*)box, 36);

    // write number of atoms to outfile
    outfile->write((char*)&atomCount, 4);

    // write precision to outfile
    float prec = precision * 10.0f;
    changeByteOrder((char*)&prec);
    outfile->write((char*)&prec, 4);

    // write maxint[] and minint[] to outfile
    int maxVal[] = {maxInt[0], maxInt[1], maxInt[2]};
    int minVal[] = {minInt[0], minInt[1], minInt[2]};
    for (i = 0; i < 3; i++) {
        changeByteOrder((char*)&maxVal[i]);
        changeByteOrder((char*)&minVal[i]);
    }
    outfile->write((char*)&minVal, 12);
    outfile->write((char*)&maxVal, 12);

    // write smallidx to outfile
    int smallidx = 0;
    outfile->write((char*)&smallidx, 4);

    unsigned int bitoffset = 0;

    byteSize = ((bitsize + 1) * (AtomCount() + 1)) / 8 + 1;

    // byteSize = actual size of the datablock
    // leave the rest filled with zeros
    char* charbuff = new char[byteSize + (4 - byteSize % 4) % 4];
    char* charPt = charbuff;

    // important for bit-operations to work properly
    memset(charbuff, 0x00, byteSize + ((4 - byteSize % 4) % 4));

    // loop through all coordinate-triplets,transform coords to
    // unsigned ints and encode
    for (i = 0; i < AtomCount(); i++) {

        thiscoord[0] = (int)(atomPosition[i * 3 + 0] * precision) - minInt[0];
        thiscoord[1] = (int)(atomPosition[i * 3 + 1] * precision) - minInt[1];
        thiscoord[2] = (int)(atomPosition[i * 3 + 2] * precision) - minInt[2];

        encodeints(charPt, bitsize, sizes, thiscoord, bitoffset);

        // update charPt
        charPt += ((bitoffset + bitsize) / 8);
        // calc new bitoffset
        bitoffset = (bitoffset + bitsize) % 8;


        // flag that runlength didn't change
        encodebits(charPt, 1, bitoffset, 0);

        // update charPt
        charPt += ((bitoffset + 1) / 8);
        // calc new bitoffset
        bitoffset = (bitoffset + 1) % 8;
    }

    // write the size to outfile
    unsigned int s = byteSize;
    changeByteOrder((char*)&s);
    outfile->write((char*)&s, 4);

    // write buffer to file
    outfile->write(charbuff, byteSize + ((4 - byteSize % 4) % 4));

    delete[] box;
    delete[] charbuff;

    return true;
}

/*
 * encode ints
 *
 */
bool GROLoader::Frame::encodeints(
    char* outbuff, int num_of_bits, unsigned int sizes[], int inbuff[], unsigned int bitoffset) {

    int i;
    unsigned int bytes[32], num_of_bytes, bytecnt, tmp;

    tmp = inbuff[0];
    num_of_bytes = 0;
    char* buffPt = outbuff;

    // interpret every byte of the three ints as an unsigned int
    do {
        bytes[num_of_bytes++] = tmp & 0xff;
        tmp >>= 8;
    } while (tmp != 0);


    // loop trough all three ints and encode
    for (i = 1; i < 3; i++) {
        if (inbuff[i] >= static_cast<int>(sizes[i])) {
            return false;
        }

        // use one step multiply
        tmp = inbuff[i];
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while (tmp != 0) {
            bytes[bytecnt++] = tmp & 0xff;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }

    // write the result in the outbuffer
    if (num_of_bits >= static_cast<int>(num_of_bytes) * 8) {
        for (i = 0; i < static_cast<int>(num_of_bytes); i++) {
            // bitsize = 8 --> offset doesn't change
            encodebits(buffPt, 8, bitoffset, bytes[i]);
            buffPt++;
        }
        encodebits(buffPt, num_of_bits - num_of_bytes * 8, bitoffset, 0);
    } else {
        for (i = 0; i < static_cast<int>(num_of_bytes - 1); i++) {
            // bitsize = 8 --> offset doesn't change
            encodebits(buffPt, 8, bitoffset, bytes[i]);
            buffPt++;
        }
        encodebits(buffPt, num_of_bits - (num_of_bytes - 1) * 8, bitoffset, bytes[i]);
    }

    return true;
}

/*
 * encode an integer to binary
 */
void GROLoader::Frame::encodebits(char* outbuff, int bitsize, int bitoffset, unsigned int num) {

    num <<= (32 - (bitoffset + bitsize));
    char* numpt = (char*)&num;

    // change byte order on little endian systems
    outbuff[0] = outbuff[0] | numpt[3];
    outbuff[1] = outbuff[1] | numpt[2];
    outbuff[2] = outbuff[2] | numpt[1];
    outbuff[3] = outbuff[3] | numpt[0];
}

/*
 * read frame-data from a given xtc-file
 */
void GROLoader::Frame::readFrame(std::fstream* file) {

    int* buffer;
    char* buffPt;
    int thiscoord[3], prevcoord[3], tempCoord;
    int run = 0;
    unsigned int i = 0;
    int bit_offset = 0;

    unsigned int sizeint[3], sizesmall[3], bitsizeint[3];
    int flag;
    int smallnum, smaller, larger, is_smaller;
    unsigned int bitsize;
    unsigned int size;

    int minint[3], maxint[3];
    int smallidx;
    float precision;

    // note that magicints[FIRSTIDX-1] == 0
    const int magicints[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203,
        256, 322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003, 16384,
        20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063, 262144, 330280, 416127, 524287,
        660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042, 8388607,
        10568983, 13316085, 16777216};

    const int FIRSTIDX = 9;
    const int LASTIDX = (sizeof(magicints) / sizeof(*magicints));


    // skip header data:
    // + version number     ( 4 Bytes)
    // + number of atoms    ( 4 Bytes)
    // + simulation step    ( 4 Bytes)
    // + simulation time    ( 4 Bytes)
    // + bounding box       (36 Bytes)
    // + number of atoms    ( 4 Bytes)
    file->seekg(56, std::ios_base::cur);


    // no compression is used for three atoms or less
    if (atomCount <= 3) {
        float posX, posY, posZ;
        for (i = 0; i < atomCount; i++) {
            file->read((char*)&posX, 4);
            changeByteOrder((char*)&posX);
            file->read((char*)&posY, 4);
            changeByteOrder((char*)&posY);
            file->read((char*)&posZ, 4);
            changeByteOrder((char*)&posZ);
            this->SetAtomPosition(i, posX, posY, posZ);
        }
        return;
    }

    // read the precision of the float coordinates
    file->read((char*)&precision, 4);
    changeByteOrder((char*)&precision);
    precision /= 10.0f;

    // read the lower bound of 'big' integer-coordinates
    file->read((char*)&minint, 12);
    changeByteOrder((char*)&minint[0]);
    changeByteOrder((char*)&minint[1]);
    changeByteOrder((char*)&minint[2]);

    // read the upper bound of 'big' integer-coordinates
    file->read((char*)&maxint, 12);
    changeByteOrder((char*)&maxint[0]);
    changeByteOrder((char*)&maxint[1]);
    changeByteOrder((char*)&maxint[2]);


    sizeint[0] = maxint[0] - minint[0] + 1;
    sizeint[1] = maxint[1] - minint[1] + 1;
    sizeint[2] = maxint[2] - minint[2] + 1;

    // check if one of the sizes is to big to be multiplied
    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        bitsize = 0; // flag the use of large sizes
    } else {
        bitsizeint[0] = 0;
        bitsizeint[1] = 0;
        bitsizeint[2] = 0;
        bitsize = sizeofints(sizeint);
    }

    // read number of bits used to encode 'small' integers
    // note: changes dynamically within one frame
    file->read((char*)&smallidx, 4);
    changeByteOrder((char*)&smallidx);

    // calculate maxidx/minidx
    int minidx, maxidx;
    if (LASTIDX < (unsigned int)(smallidx + 8)) {
        maxidx = LASTIDX;
    } else {
        maxidx = smallidx + 8;
    }
    minidx = maxidx - 8; // often this equal smallidx

    // if the difference to the last coordinate is smaller than smallnum
    // the difference is stored instead of the real coordinate
    smallnum = magicints[smallidx] / 2;

    // range of the 'small' integers
    sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];

    // calculate smaller/larger
    if (FIRSTIDX > smallidx - 1) {
        smaller = magicints[FIRSTIDX] / 2;
    } else {
        smaller = magicints[smallidx - 1] / 2;
    }
    larger = magicints[maxidx];

    // read the size of the compressed data-block
    file->read((char*)&size, 4);
    changeByteOrder((char*)&size);

    buffer = new int[(int)(atomCount * 3 * 1.2)];

    // get the compressed data-block
    file->read((char*)&buffer[0], size);

    buffPt = (char*)buffer;
    bit_offset = 0;


    while (i < atomCount) {

        thiscoord[0] = 0;
        thiscoord[1] = 0;
        thiscoord[2] = 0;


        // if large numbers are used
        if (bitsize == 0) {
            thiscoord[0] = decodebits(buffPt, bit_offset, bitsizeint[0]);
            buffPt += (bit_offset + bitsizeint[0]) / 8;
            bit_offset = (bit_offset + bitsizeint[0]) % 8;

            thiscoord[1] = decodebits(buffPt, bit_offset, bitsizeint[1]);
            buffPt += (bit_offset + bitsizeint[1]) / 8;
            bit_offset = (bit_offset + bitsizeint[1]) % 8;

            thiscoord[2] = decodebits(buffPt, bit_offset, bitsizeint[2]);
            buffPt += (bit_offset + bitsizeint[2]) / 8;
            bit_offset = (bit_offset + bitsizeint[2]) % 8;
        } else {
            decodeints(buffPt, bit_offset, bitsize, sizeint, thiscoord);
            buffPt += (bit_offset + bitsize) / 8;
            bit_offset = (bit_offset + bitsize) % 8;
        }

        // transform to unsigned ints
        thiscoord[0] += minint[0];
        thiscoord[1] += minint[1];
        thiscoord[2] += minint[2];

        // flag has been set if runlength changed while compression
        // runlength is encoded in run/3
        // is_smaller is encoded in run%3 (-1,0,1)
        flag = decodebits(buffPt, bit_offset, 1);
        buffPt += (bit_offset + 1) / 8;
        bit_offset = (bit_offset + 1) % 8;

        is_smaller = 0;
        if (flag == 1) {
            run = decodebits(buffPt, bit_offset, 5);
            buffPt += (bit_offset + 5) / 8;
            bit_offset = (bit_offset + 5) % 8;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller--;
        }

        // run = the number of coordinates following the current coordinate that
        // have bin stored as differences to there previous coordinate
        if (run > 0) {
            // save the current coordinate
            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];

            for (int k = 0; k < run; k += 3) {
                decodeints(buffPt, bit_offset, smallidx, sizesmall, thiscoord);

                buffPt += (bit_offset + smallidx) / 8;
                bit_offset = (bit_offset + smallidx) % 8;

                thiscoord[0] += prevcoord[0] - smallnum;
                thiscoord[1] += prevcoord[1] - smallnum;
                thiscoord[2] += prevcoord[2] - smallnum;

                if (k == 0) {
                    // interchange first with second atom for better
                    // compression of water molecules
                    tempCoord = thiscoord[0];
                    thiscoord[0] = prevcoord[0];
                    prevcoord[0] = tempCoord;
                    tempCoord = thiscoord[1];
                    thiscoord[1] = prevcoord[1];
                    prevcoord[1] = tempCoord;
                    tempCoord = thiscoord[2];
                    thiscoord[2] = prevcoord[2];
                    prevcoord[2] = tempCoord;

                    // calculate float-value of the old coordinate
                    this->SetAtomPosition(i, (float)prevcoord[0] / precision, (float)prevcoord[1] / precision,
                        (float)prevcoord[2] / precision);
                    i++;
                } else {
                    prevcoord[0] = thiscoord[0];
                    prevcoord[1] = thiscoord[1];
                    prevcoord[2] = thiscoord[2];
                }

                this->SetAtomPosition(i, (float)thiscoord[0] / precision, (float)thiscoord[1] / precision,
                    (float)thiscoord[2] / precision);
                i++;
            }
        } else {
            this->SetAtomPosition(
                i, (float)thiscoord[0] / precision, (float)thiscoord[1] / precision, (float)thiscoord[2] / precision);
            i++;
        }


        // update smallidx etc
        smallidx += is_smaller;
        if (is_smaller < 0) {
            smallnum = smaller;
            if (smallidx > FIRSTIDX) {
                smaller = magicints[smallidx - 1] / 2;
            } else {
                smaller = 0;
            }
        } else if (is_smaller > 0) {
            smaller = smallnum;
            smallnum = magicints[smallidx] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];
    }

    delete[] buffer;

    // set file pointer to the beginning of the next frame
    file->seekg((4 - size % 4) % 4, std::ios_base::cur);
}

/*
 * Assign a position to the array of positions.
 */
bool GROLoader::Frame::SetAtomPosition(unsigned int idx, float x, float y, float z) {
    if (idx >= this->atomCount)
        return false;
    this->atomPosition[idx * 3 + 0] = x;
    this->atomPosition[idx * 3 + 1] = y;
    this->atomPosition[idx * 3 + 2] = z;
    return true;
}

/*
 * Assign a position to the array of positions.
 */
bool GROLoader::Frame::SetAtomBFactor(unsigned int idx, float val) {
    if (idx >= this->atomCount)
        return false;
    this->bfactor[idx] = val;
    return true;
}

/*
 * Assign a charge to the array of charges.
 */
bool GROLoader::Frame::SetAtomCharge(unsigned int idx, float val) {
    if (idx >= this->atomCount)
        return false;
    this->charge[idx] = val;
    return true;
}

/*
 * Assign a occupancy to the array of occupancies.
 */
bool GROLoader::Frame::SetAtomOccupancy(unsigned int idx, float val) {
    if (idx >= this->atomCount)
        return false;
    this->occupancy[idx] = val;
    return true;
}

// ======================================================================

/*
 * protein::GROLoader::GROLoader
 */
GROLoader::GROLoader(void)
        : AnimDataModule()
        , groFilenameSlot("groFilename", "The path to the GRO data file to be loaded")
        , xtcFilenameSlot("xtcFilename", "The path to the XTC data file to be loaded")
        , forceDataCallerSlot("getforcedata", "Connects the loader with force data storage")
        , dataOutSlot("dataout", "The slot providing the loaded data")
        , maxFramesSlot("maxFrames", "The maximum number of frames to be loaded")
        , strideFlagSlot("strideFlag", "The flag wether STRIDE should be used or not.")
        , solventResidues("solventResidues", "slot to specify a ;-list of residues to be merged into separate chains")
        , mDDHostAddressSlot("mDDHostAddress", "The host address of the machine running MDDriver.")
        , mDDPortSlot("mDDPort", "The port used for data transfer on the machine running MDDriver.")
        , mDDGoSlot("mDDGo", "The toggle to pause/unpause the MDDriver simulation.")
        , mDDTransferRateSlot("mDDTransferRate", "The data transfer rate used by MDDriver.")
        ,

        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , datahash(0)
        , stride(0)
        , secStructAvailable(false)
        , numXTCFrames(0)
        , XTCFrameOffset(0)
        , xtcFileValid(false) {

    this->groFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->groFilenameSlot);

    this->xtcFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->xtcFilenameSlot);

    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &GROLoader::getData);
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &GROLoader::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->maxFramesSlot << new param::IntParam(500);
    this->MakeSlotAvailable(&this->maxFramesSlot);

    this->strideFlagSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->strideFlagSlot);

    this->solventResidues << new param::StringParam("");
    this->MakeSlotAvailable(&this->solventResidues);
    // MDDriver host name/ip
    this->mDDHostAddressSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->mDDHostAddressSlot);
    // port (default, min, max)
    this->mDDPortSlot << new param::IntParam(3000, 1024, 32767);
    this->MakeSlotAvailable(&this->mDDPortSlot);
    // go (off by default)
    this->mDDGoSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->mDDGoSlot);
    // transfer rate in ms (default, min, max)
    this->mDDTransferRateSlot << new param::IntParam(1, 1, 5000);
    this->MakeSlotAvailable(&this->mDDTransferRateSlot);

    mdd = NULL; // no mdd object
}

/*
 * protein::GROLoader::~GROLoader
 */
GROLoader::~GROLoader(void) {
    if (mdd != NULL) {
        if (mdd->IsRunning()) {
            mdd->RequestTerminate();
        }
        while (mdd->IsRunning()) {
            // wait for the thread to finish
        }
        delete mdd;
    }
    this->Release();
}

/*
 * GROLoader::create
 */
bool GROLoader::create(void) {
    // intentionally empty
    return true;
}

/*
 * GROLoader::getData
 */
bool GROLoader::getData(core::Call& call) {
    using megamol::core::utility::log::Log;

    MolecularDataCall* dc = dynamic_cast<MolecularDataCall*>(&call);
    if (dc == NULL)
        return false;

    if (this->groFilenameSlot.IsDirty() || this->solventResidues.IsDirty()) {
        this->groFilenameSlot.ResetDirty();
        this->solventResidues.ResetDirty();
        this->loadFile(this->groFilenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
    }

    if (this->mDDHostAddressSlot.IsDirty() || this->mDDPortSlot.IsDirty()) {
        this->mDDHostAddressSlot.ResetDirty();
        this->mDDPortSlot.ResetDirty();

        if (mdd != NULL) {
            if (mdd->IsRunning()) {
                mdd->RequestTerminate();
            }

            while (mdd->IsRunning()) {
                // wait for thread to terminate
            }
            delete mdd;
            mdd = NULL;
        }

        mdd = new vislib::sys::RunnableThread<MDDriverConnector>;

        ASSERT(mdd != NULL); // make sure memory was able to be allocated

        // Set the correct host and port in the mdd
        this->mdd->Initialize(this->mDDHostAddressSlot.Param<core::param::StringParam>()->Value().c_str(),
            this->mDDPortSlot.Param<core::param::IntParam>()->Value());

        // Check if simulation should be paused at the start
        if (this->mDDGoSlot.Param<core::param::BoolParam>()->Value() == false) {
            this->mdd->RequestPause(); // if so, request a pause message
        }

        // only start the thread if a valid host is provided (non-blank)
        if (!(this->mDDHostAddressSlot.Param<core::param::StringParam>()->Value()).empty()) {
            mdd->Start(); // start the mdd thread (also starts up the socket)
        }
    }

    // If mdd has been allocated, do work with it
    if (mdd != NULL) {
        // check if pause/unpause needs to be set
        if (this->mDDGoSlot.IsDirty()) {
            this->mDDGoSlot.ResetDirty();
            if (this->mDDGoSlot.Param<core::param::BoolParam>()->Value() == true) {
                // unpause the simulation (check if socket is valid first)
                if (mdd->IsSocketFunctional()) {
                    this->mdd->RequestGo();
                    // DEBUG also send the transfer rate again to see if it gets rid of the choppiness
                    this->mdd->RequestTransferRate(this->mDDTransferRateSlot.Param<core::param::IntParam>()->Value());
                }
            } else {
                // pause the simulation (check if socket is valid first)
                if (mdd->IsSocketFunctional()) {
                    this->mdd->RequestPause();
                }
            }
        }

        // update transfer rate if necessary
        if (this->mDDTransferRateSlot.IsDirty()) {
            this->mDDTransferRateSlot.ResetDirty();
            // check that socket is valid first
            if (mdd->IsSocketFunctional()) {
                this->mdd->RequestTransferRate(this->mDDTransferRateSlot.Param<core::param::IntParam>()->Value());
            }
        }

        // get data via MDDriver if socket is valid and go is on
        if (mdd->IsSocketFunctional() && this->mDDGoSlot.Param<core::param::BoolParam>()->Value() == true) {
            this->mdd->GetCoordinates(this->data[0]->AtomCount(), const_cast<float*>(this->data[0]->AtomPositions()));
        }
    } // mdd != NULL

    dc->SetDataHash(this->datahash);

    // if no xtc-filename has been set
    if (!this->xtcFileValid) {

        //if( dc->FrameID() >= this->data.Count() ) return false;
        if (dc->FrameID() >= this->data.Count()) {
            if (this->data.Count() > 0) {
                dc->SetFrameID(static_cast<unsigned int>(this->data.Count() - 1));
            } else {
                return false;
            }
        }

        dc->SetAtoms(this->data[dc->FrameID()]->AtomCount(), static_cast<unsigned int>(this->atomType.Count()),
            this->atomTypeIdx.PeekElements(), this->data[dc->FrameID()]->AtomPositions(), this->atomType.PeekElements(),
            this->atomResidueIdx.PeekElements(), this->data[dc->FrameID()]->AtomBFactor(),
            this->data[dc->FrameID()]->AtomCharge(), this->data[dc->FrameID()]->AtomOccupancy());

        dc->SetBFactorRange(this->data[dc->FrameID()]->MinBFactor(), this->data[dc->FrameID()]->MaxBFactor());
        dc->SetChargeRange(this->data[dc->FrameID()]->MinCharge(), this->data[dc->FrameID()]->MaxCharge());
        dc->SetOccupancyRange(this->data[dc->FrameID()]->MinOccupancy(), this->data[dc->FrameID()]->MaxOccupancy());
    } else {

        Frame* fr = NULL;
        fr = dynamic_cast<GROLoader::Frame*>(this->requestLockedFrame(dc->FrameID()));

        if (fr == NULL)
            return false;

        dc->SetUnlocker(new Unlocker(*fr));

        dc->SetAtoms(this->data[0]->AtomCount(), static_cast<unsigned int>(this->atomType.Count()),
            this->atomTypeIdx.PeekElements(), fr->AtomPositions(), this->atomType.PeekElements(),
            this->atomResidueIdx.PeekElements(), this->data[0]->AtomBFactor(), this->data[0]->AtomCharge(),
            this->data[0]->AtomOccupancy());

        dc->SetBFactorRange(this->data[0]->MinBFactor(), this->data[0]->MaxBFactor());
        dc->SetChargeRange(this->data[0]->MinCharge(), this->data[0]->MaxCharge());
        dc->SetOccupancyRange(this->data[0]->MinOccupancy(), this->data[0]->MaxOccupancy());
    }

    dc->SetConnections(
        static_cast<unsigned int>(this->connectivity.Count() / 2), (unsigned int*)this->connectivity.PeekElements());
    dc->SetResidues(static_cast<unsigned int>(this->residue.Count()),
        (const MolecularDataCall::Residue**)this->residue.PeekElements());
    // dc->SetAtomResidueIndices(this->atomResidueIdx.PeekElements());
    dc->SetSolventResidueIndices(
        static_cast<unsigned int>(this->solventResidueIdx.Count()), this->solventResidueIdx.PeekElements());
    dc->SetResidueTypeNames(static_cast<unsigned int>(this->residueTypeName.Count()),
        (vislib::StringA*)this->residueTypeName.PeekElements());
    dc->SetMolecules(
        static_cast<unsigned int>(this->molecule.Count()), (MolecularDataCall::Molecule*)this->molecule.PeekElements());
    dc->SetChains(
        static_cast<unsigned int>(this->chain.Count()), (MolecularDataCall::Chain*)this->chain.PeekElements());

    if (!this->secStructAvailable && this->strideFlagSlot.Param<param::BoolParam>()->Value()) {
        time_t t = clock(); // DEBUG
        if (this->stride)
            delete this->stride;
        this->stride = new Stride(dc);
        this->stride->WriteToInterface(dc);
        this->secStructAvailable = true;
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Secondary Structure computed via STRIDE in %f seconds.",
            (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG
    } else if (this->strideFlagSlot.Param<param::BoolParam>()->Value()) {
        this->stride->WriteToInterface(dc);
    }

    // Set the filter array for the molecular data call
    dc->SetFilter(this->atomVisibility.PeekElements());

    return true;
}

/*
 * GROLoader::getExtent
 */
bool GROLoader::getExtent(core::Call& call) {
    MolecularDataCall* dc = dynamic_cast<MolecularDataCall*>(&call);
    if (dc == NULL)
        return false;

    if (this->groFilenameSlot.IsDirty() || this->solventResidues.IsDirty()) {
        this->groFilenameSlot.ResetDirty();
        this->solventResidues.ResetDirty();
        this->loadFile(this->groFilenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
    }

    if (this->mDDHostAddressSlot.IsDirty() || this->mDDPortSlot.IsDirty()) {
        this->mDDHostAddressSlot.ResetDirty();
        this->mDDPortSlot.ResetDirty();

        if (mdd != NULL) {
            if (mdd->IsRunning()) {
                mdd->RequestTerminate();
            }

            while (mdd->IsRunning()) {
                // wait for thread to terminate
            }
            delete mdd;
            mdd = NULL;
        }

        mdd = new vislib::sys::RunnableThread<MDDriverConnector>;

        ASSERT(mdd != NULL); // make sure memory was able to be allocated

        // Set the correct host and port in the mdd
        this->mdd->Initialize(this->mDDHostAddressSlot.Param<core::param::StringParam>()->Value().c_str(),
            this->mDDPortSlot.Param<core::param::IntParam>()->Value());

        // Check if simulation should be paused at the start
        if (this->mDDGoSlot.Param<core::param::BoolParam>()->Value() == false) {
            this->mdd->RequestPause(); // if so, request a pause message
        }

        if (!(this->mDDHostAddressSlot.Param<core::param::StringParam>()->Value()).empty()) {
            mdd->Start(); // start the mdd thread (also starts up the socket)
        }
    }

    // Only deal with mdd if it is valid
    if (mdd != NULL) {
        // check if pause/unpause needs to be set
        if (this->mDDGoSlot.IsDirty()) {
            this->mDDGoSlot.ResetDirty();
            if (this->mDDGoSlot.Param<core::param::BoolParam>()->Value() == true) {
                // unpause the simulation (check if socket is valid first)
                if (mdd->IsSocketFunctional()) {
                    this->mdd->RequestGo();
                    // Also send transfer rate again otherwise the simulation is choppy
                    this->mdd->RequestTransferRate(this->mDDTransferRateSlot.Param<core::param::IntParam>()->Value());
                }
            } else {
                // pause the simulation (check if socket is valid first)
                if (mdd->IsSocketFunctional()) {
                    this->mdd->RequestPause();
                }
            }
        }

        // update transfer rate if necessary
        if (this->mDDTransferRateSlot.IsDirty()) {
            this->mDDTransferRateSlot.ResetDirty();
            // check that socket is valid first
            if (mdd->IsSocketFunctional()) {
                this->mdd->RequestTransferRate(this->mDDTransferRateSlot.Param<core::param::IntParam>()->Value());
            }
        }
    } // mdd != NULL

    // grow bounding box by 3.0 Angstrom (for volume rendering / SAS)
    vislib::math::Cuboid<float> bBoxPlus3(this->bbox);
    bBoxPlus3.Grow(3.0f);

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox(bBoxPlus3);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox(bBoxPlus3);

    if (!xtcFileValid) {
        // no XTC file set or loaded --> use number of loaded frames
        dc->SetFrameCount(vislib::math::Max(1U, static_cast<unsigned int>(this->data.Count())));
    } else {
        // XTC file set and loaded --> use number of frames
        dc->SetFrameCount(vislib::math::Max(1U, static_cast<unsigned int>(this->numXTCFrames)));
    }

    dc->SetDataHash(this->datahash);

    return true;
}

/*
 * GROLoader::release
 */
void GROLoader::release(void) {
    // stop frame-loading thread before clearing data array
    resetFrameCache();

    for (int i = 0; i < (int)this->data.Count(); i++)
        delete data[i];
    this->data.Clear();

    for (int i = 0; i < (int)this->residue.Count(); i++)
        delete residue[i];
    this->residue.Clear();

    delete stride;
}


/*
 * GROLoader::constructFrame
 */
view::AnimDataModule::Frame* GROLoader::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<GROLoader*>(this));
    f->SetAtomCount(this->data[0]->AtomCount());
    return f;
}

/*
 * GROLoader::loadFrame
 */
void GROLoader::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {

    //time_t t = clock();

    GROLoader::Frame* fr = dynamic_cast<GROLoader::Frame*>(frame);
    //int atomCnt = this->data[0]->AtomCount();

    // set the frames index
    fr->setFrameIdx(idx);

    // read first frame from the existing data-buffer
    /*if( idx == 0 ) {
        for( unsigned int i = 0; i < atomCnt*3; i+=3 ) {
            fr->SetAtomPosition(i/3, data[0]->AtomPositions()[i],
                                data[0]->AtomPositions()[i+1],
                                data[0]->AtomPositions()[i+2]);
        }
    } else {*/
    std::fstream xtcFile;

    xtcFile.open(this->xtcFilenameSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);

    xtcFile.seekg(this->XTCFrameOffset[idx]);

    fr->readFrame(&xtcFile);

    xtcFile.close();
    //}

    //megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO,
    //"Time for loading frame %i: %f", idx,
    //( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG
}

/*
 * GROLoader::loadFile
 */
void GROLoader::loadFile(const vislib::TString& filename) {
    using megamol::core::utility::log::Log;

    this->resetAllData();

    this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < (int)this->data.Count(); i++)
        delete data[i];
    this->data.Clear();

    // stop frame-loading thread if neccessary
    if (xtcFileValid)
        resetFrameCache();

    this->data.Clear();
    this->datahash++;

    time_t t = clock(); // DEBUG

    vislib::StringA line;
    unsigned int idx, atomCnt, totalAtomCnt, frameCnt, resCnt, chainCnt;

    t = clock(); // DEBUG

    vislib::sys::ASCIIFileBuffer file;
    SIZE_T frameCapacity = 10000;

    // try to load the file
    if (file.LoadFile(T2A(filename))) {
        // file successfully loaded
        if (file.Count() > 2) {
            // read number of atoms
            totalAtomCnt = atoi(file[1]);
        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Atom count: %i", totalAtomCnt); // DEBUG

        // Init atom filter array with 1 (= 'visible')
        if (!this->atomVisibility.IsEmpty())
            this->atomVisibility.Clear(true);
        this->atomVisibility.SetCount(totalAtomCnt);
        for (unsigned int at = 0; at < totalAtomCnt; at++)
            this->atomVisibility[at] = 1;

        // set the atom count for the first frame
        frameCnt = 0;
        this->data.AssertCapacity(frameCapacity);
        this->data.SetCount(1);
        this->data[0] = new Frame(*const_cast<GROLoader*>(this));
        this->data[0]->SetAtomCount(static_cast<unsigned int>(totalAtomCnt));
        this->data[0]->setFrameIdx(0);
        // resize atom type index array
        this->atomTypeIdx.SetCount(totalAtomCnt);
        // set the capacity of the atom type array
        this->atomType.AssertCapacity(totalAtomCnt);
        // set the capacity of the residue array
        this->residue.AssertCapacity(totalAtomCnt);

        this->atomResidueIdx.SetCount(totalAtomCnt);

        // check for residue-parameter and make it a chain of its own ( if no chain-id is specified ...?)
        const vislib::TString& solventResiduesStr =
            this->solventResidues.Param<core::param::StringParam>()->Value().c_str();
        // get all the solvent residue names to filter out
        vislib::Array<vislib::TString> solventResidueNames =
            vislib::StringTokeniser<vislib::TCharTraits>::Split(solventResiduesStr, ';', true);
        //this->solventResidueIdx.SetCount(solventResidueNames);
        //memset(&this->solventResidueIdx[0], -1, this->solventResidueIdx.Count()*sizeof(int));
        this->solventResidueIdx.Clear();

        // parse all atoms
        vislib::StringA line;
        for (atomCnt = 0; atomCnt < totalAtomCnt; ++atomCnt) {
            line = file[atomCnt + 2];
            this->parseAtomEntry(line, atomCnt, frameCnt, solventResidueNames);
        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time for parsing first frame: %f",
            (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG

        this->molecule.AssertCapacity(this->residue.Count());
        //this->chain.AssertCapacity( this->residue.Count()); ?????

        unsigned int first, cnt;

        unsigned int firstConIdx;
        // loop over all chains
        this->chain.AssertCapacity(this->chainFirstRes.Count());
        for (chainCnt = 0; chainCnt < this->chainFirstRes.Count(); ++chainCnt) {
            // add new molecule
            if (chainCnt == 0) {
                this->molecule.Add(MolecularDataCall::Molecule(0, 1, chainCnt));
                firstConIdx = 0;
            } else {
                this->molecule.Add(MolecularDataCall::Molecule(
                    this->molecule.Last().FirstResidueIndex() + this->molecule.Last().ResidueCount(), 1, chainCnt));
                firstConIdx = static_cast<unsigned int>(this->connectivity.Count());
            }
            // add new chain
            this->chain.Add(MolecularDataCall::Chain(static_cast<unsigned int>(this->molecule.Count() - 1), 1,
                this->chainName[chainCnt], this->chainType[chainCnt]));
            // get the residue range of the current chain
            first = this->chainFirstRes[chainCnt];
            cnt = first + this->chainResCount[chainCnt];
            // loop over all residues in the current chain
            for (resCnt = first; resCnt < cnt; ++resCnt) {
                this->residue[resCnt]->SetMoleculeIndex(static_cast<unsigned int>(this->molecule.Count() - 1));
                // search for connections inside the current residue
                this->MakeResidueConnections(resCnt, 0);
                // search for connections between consecutive residues
                if ((resCnt + 1) < cnt) {
                    if (this->MakeResidueConnections(resCnt, resCnt + 1, 0)) {
                        this->molecule.Last().SetPosition(
                            this->molecule.Last().FirstResidueIndex(), this->molecule.Last().ResidueCount() + 1);
                    } else {
                        this->molecule.Last().SetConnectionRange(
                            firstConIdx, (static_cast<unsigned int>(this->connectivity.Count()) - firstConIdx) / 2);
                        firstConIdx = static_cast<unsigned int>(this->connectivity.Count());
                        this->molecule.Add(MolecularDataCall::Molecule(resCnt + 1, 1, chainCnt));
                        this->chain.Last().SetPosition(
                            this->chain.Last().FirstMoleculeIndex(), this->chain.Last().MoleculeCount() + 1);
                    }
                }
            }
            this->molecule.Last().SetConnectionRange(
                firstConIdx, (static_cast<unsigned int>(this->connectivity.Count()) - firstConIdx) / 2);
        }
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "Time for finding all bonds: %f", (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG

        // search for CA, C, O and N in amino acids
        MolecularDataCall::AminoAcid* aminoacid;
        for (resCnt = 0; resCnt < this->residue.Count(); ++resCnt) {
            // check if the current residue is an amino acid
            if (this->residue[resCnt]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
                aminoacid = (MolecularDataCall::AminoAcid*)this->residue[resCnt];
                idx = aminoacid->FirstAtomIndex();
                cnt = idx + aminoacid->AtomCount();
                // loop over all atom of the current amino acid
                for (atomCnt = idx; atomCnt < cnt; ++atomCnt) {
                    if (this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals("CA")) {
                        aminoacid->SetCAlphaIndex(atomCnt);
                    } else if (this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals("N")) {
                        aminoacid->SetNIndex(atomCnt);
                    } else if (this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals("C")) {
                        aminoacid->SetCCarbIndex(atomCnt);
                    } else if (this->atomType[this->atomTypeIdx[atomCnt]].Name().Equals("O")) {
                        aminoacid->SetOIndex(atomCnt);
                    }
                }
            }
        }

        //Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for loading file %s: %f", T2A( filename), ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "Time for loading file: %f", (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG


        // if xtc-filename has been set
        if (!this->xtcFilenameSlot.Param<core::param::FilePathParam>()->Value().empty()) {
            // try to get the total number of frames and calculate the
            // bounding box
            this->readNumXTCFrames();

            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Number of XTC-frames: %u", this->numXTCFrames); // DEBUG

            //float box[3][3];
            char tmpByte;
            std::fstream xtcFile;
            char* num;
            unsigned int nAtoms;

            // try to open the xtc-file
            xtcFile.open(
                this->xtcFilenameSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);

            if (!xtcFile) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Could not load XTC-file."); // DEBUG
                xtcFileValid = false;
            } else {


                xtcFile.seekg(4, std::ios_base::cur);
                // read number of atoms
                xtcFile.read((char*)&nAtoms, 4);
                // change byte order
                num = (char*)&nAtoms;
                tmpByte = num[0];
                num[0] = num[3];
                num[3] = tmpByte;
                tmpByte = num[1];
                num[1] = num[2];
                num[2] = tmpByte;

                // check whether the pdb-file and the xtc-file contain the
                // same number of atoms
                if (nAtoms != totalAtomCnt) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "XTC-File and given PDB-file not matching (XTC-file has"
                        "%i atom entries, PDB-file has %i atom entries).",
                        nAtoms, totalAtomCnt); // DEBUG
                    xtcFileValid = false;
                    xtcFile.close();
                } else {
                    xtcFile.close();

                    xtcFileValid = true;

                    int maxFrames = vislib::math::Min<int>(this->maxFramesSlot.Param<core::param::IntParam>()->Value(),
                        static_cast<int>(this->numXTCFrames));

                    // frames in xtc-file - 1 (without the last frame)
                    this->setFrameCount(this->numXTCFrames);

                    // start the loading thread
                    this->initFrameCache(maxFrames);
                }
            }
        }


    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Could not load file %s", (const char*)T2A(filename)); // DEBUG
    }
}

/*
 * parse one atom entry
 */
void GROLoader::parseAtomEntry(vislib::StringA& atomEntry, unsigned int atom, unsigned int frame,
    vislib::Array<vislib::TString>& solventResidueNames) {
    // temp variables
    vislib::StringA tmpStr;
    vislib::math::Vector<float, 3> pos;
    // set atom position
    pos.Set(float(atof(atomEntry.Substring(20, 8))), float(atof(atomEntry.Substring(28, 8))),
        float(atof(atomEntry.Substring(36, 8))));
    // TOOD: do we really need the nm to Angstrom conversion?
    //pos *= 10.0f;
    this->data[frame]->SetAtomPosition(atom, pos.X(), pos.Y(), pos.Z());

    // get the name (atom type) of the current ATOM entry
    tmpStr = atomEntry.Substring(10, 5);
    tmpStr.TrimSpaces();
    // get the radius of the element
    float radius = getElementRadius(tmpStr);
    // get the color of the element
    vislib::math::Vector<unsigned char, 3> color = getElementColor(tmpStr);
    // set the new atom type
    MolecularDataCall::AtomType type(tmpStr, radius, color.X(), color.Y(), color.Z());
    // search for current atom type in atom type array
    INT_PTR atomTypeIdx = atomType.IndexOf(type);
    if (atomTypeIdx == vislib::Array<MolecularDataCall::AtomType>::INVALID_POS) {
        this->atomTypeIdx[atom] = static_cast<unsigned int>(this->atomType.Count());
        this->atomType.Add(type);
    } else {
        this->atomTypeIdx[atom] = static_cast<unsigned int>(atomTypeIdx);
    }

    // update the bounding box
    vislib::math::Cuboid<float> atomBBox(pos.X() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Y() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Z() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.X() + this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Y() + this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Z() + this->atomType[this->atomTypeIdx[atom]].Radius());
    if (atom == 0) {
        this->bbox = atomBBox;
    } else {
        this->bbox.Union(atomBBox);
    }

    // get chain id
    char tmpChainId = 0;
    MolecularDataCall::Chain::ChainType tmpChainType = MolecularDataCall::Chain::UNSPECIFIC;
    // get the name of the residue
    tmpStr = atomEntry.Substring(5, 5);
    tmpStr.TrimSpaces();
    vislib::StringA resName = tmpStr;
    unsigned int resTypeIdx;

    // search for current residue type name in the array
    INT_PTR resTypeNameIdx = this->residueTypeName.IndexOf(resName);
    if (resTypeNameIdx == vislib::Array<vislib::StringA>::INVALID_POS) {
        resTypeIdx = static_cast<unsigned int>(this->residueTypeName.Count());
        this->residueTypeName.Add(resName);

        // check if the name of the residue is matched by one of the solvent residue names
        for (unsigned int filterCnt = 0; filterCnt < solventResidueNames.Count(); ++filterCnt) {
            if (resName.StartsWithInsensitive(solventResidueNames[filterCnt])) {
                tmpChainId = SOLVENT_CHAIN_IDENTIFIER;
                tmpChainType = MolecularDataCall::Chain::SOLVENT;
                this->solventResidueIdx.Add(resTypeIdx);
                break;
            }
        }
    } else {
        resTypeIdx = static_cast<unsigned int>(resTypeNameIdx);

        // check if the index of the residue is matched by one of the existent solvent residue indices
        for (unsigned int srIdx = 0; srIdx < this->solventResidueIdx.Count(); ++srIdx) {
            if (this->solventResidueIdx[srIdx] == resTypeIdx) {
                tmpChainId = SOLVENT_CHAIN_IDENTIFIER;
                tmpChainType = MolecularDataCall::Chain::SOLVENT;
                break;
            }
        }
    }


    // get the sequence number of the residue
    tmpStr = atomEntry.Substring(0, 5);
    tmpStr.TrimSpaces();
    unsigned int newResSeq = static_cast<unsigned int>(atoi(tmpStr));
    // handle residue
    if (this->residue.Count() == 0) {
        // create first residue
        this->resSeq = newResSeq;
        if (this->IsAminoAcid(resName)) {
            MolecularDataCall::AminoAcid* res =
                new MolecularDataCall::AminoAcid(atom, 1, 0, 0, 0, 0, atomBBox, resTypeIdx, -1, newResSeq);
            this->residue.Add((MolecularDataCall::Residue*)res);
        } else {
            MolecularDataCall::Residue* res =
                new MolecularDataCall::Residue(atom, 1, atomBBox, resTypeIdx, -1, newResSeq);
            this->residue.Add(res);
        }
        // first chain
        this->chainId = tmpChainId;
        this->chainFirstRes.Clear();
        this->chainFirstRes.SetCapacityIncrement(100);
        this->chainFirstRes.Add(0);
        this->chainResCount.Clear();
        this->chainResCount.SetCapacityIncrement(100);
        this->chainResCount.Add(1);
        this->chainName.Clear();
        this->chainName.SetCapacityIncrement(100);
        this->chainName.Add(this->chainId);
        this->chainType.Add(tmpChainType);
    } else if (newResSeq == this->resSeq) {
        // still the same residue - add one atom
        this->residue.Last()->SetPosition(
            this->residue.Last()->FirstAtomIndex(), this->residue.Last()->AtomCount() + 1);
        // compute and set the bounding box
        vislib::math::Cuboid<float> resBBox(this->residue.Last()->BoundingBox());
        resBBox.Union(atomBBox);
        this->residue.Last()->SetBoundingBox(resBBox);
    } else if (newResSeq != this->resSeq) {
        // starting new residue
        this->resSeq = newResSeq;
        if (this->IsAminoAcid(resName)) {
            MolecularDataCall::AminoAcid* res =
                new MolecularDataCall::AminoAcid(atom, 1, 0, 0, 0, 0, atomBBox, resTypeIdx, -1, newResSeq);
            this->residue.Add((MolecularDataCall::Residue*)res);
        } else {
            MolecularDataCall::Residue* res =
                new MolecularDataCall::Residue(atom, 1, atomBBox, resTypeIdx, -1, newResSeq);
            this->residue.Add(res);
        }
        // elongate existing chain or create new chain
        if (tmpChainId == this->chainId) {
            this->chainResCount.Last()++;
        } else {
            this->chainId = tmpChainId;
            this->chainFirstRes.Add(static_cast<unsigned int>(this->residue.Count() - 1));
            this->chainResCount.Add(1);
            this->chainType.Add(tmpChainType);
            this->chainName.Add(this->chainId);
        }
    }
    this->atomResidueIdx[atom] = static_cast<int>(this->residue.Count() - 1);

    // get the temperature factor (b-factor)
    float tempFactor = 0.0f;
    if (atom == 0) {
        this->data[frame]->SetBFactorRange(tempFactor, tempFactor);
    } else {
        if (this->data[frame]->MinBFactor() > tempFactor)
            this->data[frame]->SetMinBFactor(tempFactor);
        else if (this->data[frame]->MaxBFactor() < tempFactor)
            this->data[frame]->SetMaxBFactor(tempFactor);
    }
    this->data[frame]->SetAtomBFactor(atom, tempFactor);

    // get the occupancy
    float occupancy = 0.0f;
    this->data[frame]->SetAtomOccupancy(atom, occupancy);

    // get the charge
    float charge = 0.0f;
    if (atom == 0) {
        this->data[frame]->SetChargeRange(charge, charge);
    } else {
        if (this->data[frame]->MinCharge() > charge)
            this->data[frame]->SetMinCharge(charge);
        else if (this->data[frame]->MaxCharge() < charge)
            this->data[frame]->SetMaxCharge(charge);
    }
    this->data[frame]->SetAtomCharge(atom, charge);
}

/*
 * Get the radius of the element
 */
float GROLoader::getElementRadius(vislib::StringA name, float scaleFactor) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while (vislib::CharTraitsA::IsDigit(name[cnt])) {
        cnt++;
    }

    // --- van der Waals radii ---
    if (name[cnt] == 'H')
        return 0.12f * scaleFactor;
    if (name[cnt] == 'C')
        return 0.17f * scaleFactor;
    if (name[cnt] == 'N')
        return 0.155f * scaleFactor;
    if (name[cnt] == 'O')
        return 0.152f * scaleFactor;
    if (name[cnt] == 'S')
        return 0.18f * scaleFactor;
    if (name[cnt] == 'P')
        return 0.18f * scaleFactor;
    if (name[cnt] == 'C')
        return 0.17f * scaleFactor;

    return 0.15f * scaleFactor;
}

/*
 * Get the color of the element
 */
vislib::math::Vector<unsigned char, 3> GROLoader::getElementColor(vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while (vislib::CharTraitsA::IsDigit(name[cnt])) {
        cnt++;
    }
    if (name[cnt] == 'H') // white or light grey
        return vislib::math::Vector<unsigned char, 3>(240, 240, 240);
    if (name[cnt] == 'C') // (dark) grey or green
        return vislib::math::Vector<unsigned char, 3>(90, 90, 90);
    //return vislib::math::Vector<unsigned char, 3>( 125, 125, 125);
    //return vislib::math::Vector<unsigned char, 3>( 90, 175, 50);
    if (name[cnt] == 'N') // blue
        //return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
        return vislib::math::Vector<unsigned char, 3>(37, 136, 195);
    if (name[cnt] == 'O') // red
        //return vislib::math::Vector<unsigned char, 3>( 250, 94, 82);
        return vislib::math::Vector<unsigned char, 3>(206, 34, 34);
    if (name[cnt] == 'S') // yellow
        //return vislib::math::Vector<unsigned char, 3>( 250, 230, 50);
        return vislib::math::Vector<unsigned char, 3>(255, 215, 0);
    if (name[cnt] == 'P') // orange
        return vislib::math::Vector<unsigned char, 3>(255, 128, 64);
    if (name[cnt] == 'M' /*&& name[cnt+1] == 'e'*/) // Methanol? -> same as carbon ...
        return vislib::math::Vector<unsigned char, 3>(90, 90, 90);
    /*
    if( name[cnt] == 'H' ) // white or light grey
        return vislib::math::Vector<unsigned char, 3>( 240, 240, 240);
    if( name[cnt] == 'C' ) // (dark) grey
        return vislib::math::Vector<unsigned char, 3>( 175, 175, 175);
    if( name[cnt] == 'N' ) // blue
        return vislib::math::Vector<unsigned char, 3>( 40, 160, 220);
    if( name[cnt] == 'O' ) // red
        return vislib::math::Vector<unsigned char, 3>( 230, 50, 50);
    if( name[cnt] == 'S' ) // yellow
        return vislib::math::Vector<unsigned char, 3>( 255, 215, 0);
    if( name[cnt] == 'P' ) // orange
        return vislib::math::Vector<unsigned char, 3>( 255, 128, 64);
    */

    return vislib::math::Vector<unsigned char, 3>(191, 191, 191);
}

/*
 * set the position of the current atom entry to the frame
 */
void GROLoader::setAtomPositionToFrame(vislib::StringA& atomEntry, unsigned int atom, unsigned int frame) {
    // temp variables
    vislib::StringA tmpStr;
    vislib::math::Vector<float, 3> pos;
    // set atom position
    pos.Set(float(atof(atomEntry.Substring(30, 8))), float(atof(atomEntry.Substring(38, 8))),
        float(atof(atomEntry.Substring(46, 8))));
    this->data[frame]->SetAtomPosition(atom, pos.X(), pos.Y(), pos.Z());

    // update bounding box
    vislib::math::Cuboid<float> atomBBox(pos.X() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Y() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Z() - this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.X() + this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Y() + this->atomType[this->atomTypeIdx[atom]].Radius(),
        pos.Z() + this->atomType[this->atomTypeIdx[atom]].Radius());
    this->bbox.Union(atomBBox);

    // get the temperature factor (b-factor)
    tmpStr = atomEntry.Substring(60, 6);
    tmpStr.TrimSpaces();
    float tempFactor = float(atof(tmpStr));
    if (atom == 0) {
        this->data[frame]->SetBFactorRange(tempFactor, tempFactor);
    } else {
        if (this->data[frame]->MinBFactor() > tempFactor)
            this->data[frame]->SetMinBFactor(tempFactor);
        else if (this->data[frame]->MaxBFactor() < tempFactor)
            this->data[frame]->SetMaxBFactor(tempFactor);
    }

    // get the occupancy
    tmpStr = atomEntry.Substring(54, 6);
    tmpStr.TrimSpaces();
    float occupancy = float(atof(tmpStr));
    if (atom == 0) {
        this->data[frame]->SetOccupancyRange(occupancy, occupancy);
    } else {
        if (this->data[frame]->MinOccupancy() > occupancy)
            this->data[frame]->SetMinOccupancy(occupancy);
        else if (this->data[frame]->MaxOccupancy() < occupancy)
            this->data[frame]->SetMaxOccupancy(occupancy);
    }

    // get the charge
    tmpStr = atomEntry.Substring(78, 2);
    tmpStr.TrimSpaces();
    float charge = float(atof(tmpStr));
    if (atom == 0) {
        this->data[frame]->SetChargeRange(charge, charge);
    } else {
        if (this->data[frame]->MinCharge() > charge)
            this->data[frame]->SetMinCharge(charge);
        else if (this->data[frame]->MaxCharge() < charge)
            this->data[frame]->SetMaxCharge(charge);
    }
}

/*
 * Search for connections in the given residue and add them to the
 * global connection array.
 */
void GROLoader::MakeResidueConnections(unsigned int resIdx, unsigned int frame) {
    // check bounds
    if (resIdx >= this->residue.Count())
        return;
    if (frame >= this->data.Count())
        return;
    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if (this->connectivity.Count() == this->connectivity.Capacity()) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity(connectionCapacity);
    }
    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for (cnt0 = 0; cnt0 < this->residue[resIdx]->AtomCount() - 1; ++cnt0) {
        for (cnt1 = cnt0 + 1; cnt1 < this->residue[resIdx]->AtomCount(); ++cnt1) {
            // get atom indices
            atomIdx0 = this->residue[resIdx]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set(this->data[frame]->AtomPositions()[3 * atomIdx0 + 0],
                this->data[frame]->AtomPositions()[3 * atomIdx0 + 1],
                this->data[frame]->AtomPositions()[3 * atomIdx0 + 2]);
            atomPos1.Set(this->data[frame]->AtomPositions()[3 * atomIdx1 + 0],
                this->data[frame]->AtomPositions()[3 * atomIdx1 + 1],
                this->data[frame]->AtomPositions()[3 * atomIdx1 + 2]);
            // check distance
            if ((atomPos0 - atomPos1).Length() < 0.58f * (this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                                                             this->atomType[this->atomTypeIdx[atomIdx1]].Radius())) {
                // add connection
                this->connectivity.Add(atomIdx0);
                this->connectivity.Add(atomIdx1);
            }
        }
    }
}

/*
 * Search for connections between two residues.
 */
bool GROLoader::MakeResidueConnections(unsigned int resIdx0, unsigned int resIdx1, unsigned int frame) {
    // flag wether the two residues are connected
    bool connected = false;
    // check bounds
    if (resIdx0 >= this->residue.Count())
        return connected;
    if (resIdx1 >= this->residue.Count())
        return connected;
    if (frame >= this->data.Count())
        return connected;

    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if (this->connectivity.Count() == this->connectivity.Capacity()) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity(connectionCapacity);
    }

    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for (cnt0 = 0; cnt0 < this->residue[resIdx0]->AtomCount(); ++cnt0) {
        for (cnt1 = 0; cnt1 < this->residue[resIdx1]->AtomCount(); ++cnt1) {
            // get atom indices
            atomIdx0 = this->residue[resIdx0]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx1]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set(this->data[frame]->AtomPositions()[3 * atomIdx0 + 0],
                this->data[frame]->AtomPositions()[3 * atomIdx0 + 1],
                this->data[frame]->AtomPositions()[3 * atomIdx0 + 2]);
            atomPos1.Set(this->data[frame]->AtomPositions()[3 * atomIdx1 + 0],
                this->data[frame]->AtomPositions()[3 * atomIdx1 + 1],
                this->data[frame]->AtomPositions()[3 * atomIdx1 + 2]);
            // check distance
            if ((atomPos0 - atomPos1).Length() < 0.58f * (this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                                                             this->atomType[this->atomTypeIdx[atomIdx1]].Radius())) {
                // add connection
                this->connectivity.Add(atomIdx0);
                this->connectivity.Add(atomIdx1);
                connected = true;
            }
        }
    }
    return connected;
}

/*
 * Check if the residue is an amino acid.
 */
bool GROLoader::IsAminoAcid(vislib::StringA resName) {
    if (resName.Equals("ALA") | resName.Equals("ARG") | resName.Equals("ASN") | resName.Equals("ASP") |
        resName.Equals("CYS") | resName.Equals("GLN") | resName.Equals("GLU") | resName.Equals("GLY") |
        resName.Equals("HIS") | resName.Equals("ILE") | resName.Equals("LEU") | resName.Equals("LYS") |
        resName.Equals("MET") | resName.Equals("PHE") | resName.Equals("PRO") | resName.Equals("SER") |
        resName.Equals("THR") | resName.Equals("TRP") | resName.Equals("TYR") | resName.Equals("VAL") |
        resName.Equals("ASH") | resName.Equals("CYX") | resName.Equals("CYM") | resName.Equals("GLH") |
        resName.Equals("HID") | resName.Equals("HIE") | resName.Equals("HIP") | resName.Equals("LYN") |
        resName.Equals("TYM"))
        return true;
    return false;
}

/*
 * reset all data containers.
 */
void GROLoader::resetAllData() {
    unsigned int cnt;
    //this->data.Clear();
    this->atomTypeIdx.Clear();
    this->atomResidueIdx.Clear();
    this->atomType.Clear();
    for (cnt = 0; cnt < this->residue.Count(); ++cnt) {
        delete this->residue[cnt];
    }
    this->residue.Clear();
    this->residueTypeName.Clear();
    this->molecule.Clear();
    this->chain.Clear();
    this->connectivity.Clear();
    delete stride;
    this->stride = 0;
    secStructAvailable = false;
    this->chainFirstRes.Clear();
    this->chainResCount.Clear();
    this->chainName.Clear();
    this->chainType.Clear();
}


/*
 * Read the number of frames from the XTC file and update the bounding box.
 * The Last frame contains wrong byte ordering and therefore gets ignored.
 */
bool GROLoader::readNumXTCFrames() {

    time_t t = clock();

    // reset values
    this->numXTCFrames = 0;
    this->XTCFrameOffset.Clear();

    // try to open xtc file
    std::fstream xtcFile;
    xtcFile.open(this->xtcFilenameSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);

    // check if file could be opened
    if (!xtcFile)
        return false;

    this->XTCFrameOffset.SetCapacityIncrement(1000);
    int size;
    char tmpByte;
    char* num;

    int minint[3];
    int maxint[3];

    unsigned int i;

    float precision;

    xtcFile.seekg(0, std::ios_base::beg);

    vislib::math::Cuboid<float> tmpBBox(this->bbox);

    // read until eof
    while (!xtcFile.eof()) {

        // add the offset to the offset array
        this->XTCFrameOffset.Add((unsigned int)xtcFile.tellg());

        // skip some header data
        xtcFile.seekg(56, std::ios_base::cur);
        // read precision
        xtcFile.read((char*)&precision, 4);
        // change byte-order
        num = (char*)&precision;
        tmpByte = num[0];
        num[0] = num[3];
        num[3] = tmpByte;
        tmpByte = num[1];
        num[1] = num[2];
        num[2] = tmpByte;
        precision /= 10.0f;

        // get the lower bound
        xtcFile.read((char*)&minint, 12);
        // get the upper bound
        xtcFile.read((char*)&maxint, 12);
        // change byte-order
        for (i = 0; i < 3; i++) {
            num = (char*)&minint[i];
            tmpByte = num[0];
            num[0] = num[3];
            num[3] = tmpByte;
            tmpByte = num[1];
            num[1] = num[2];
            num[2] = tmpByte;
            num = (char*)&maxint[i];
            tmpByte = num[0];
            num[0] = num[3];
            num[3] = tmpByte;
            tmpByte = num[1];
            num[1] = num[2];
            num[2] = tmpByte;
        }

        // update the bounding box by uniting it with the last frames box
        this->bbox.Union(tmpBBox);
        // get the current frames bounding box including the atom radius
        // note: atom radius is divided by 10
        tmpBBox = vislib::math::Cuboid<float>((float)minint[0] / precision - 0.3f, (float)minint[1] / precision - 0.3f,
            (float)minint[2] / precision - 0.3f,

            (float)maxint[0] / precision + 0.3f, (float)maxint[1] / precision + 0.3f,
            (float)maxint[2] / precision + 0.3f);

        // skip some header data
        xtcFile.seekg(4, std::ios_base::cur);

        // read size of the compressed block of data
        xtcFile.read((char*)&size, 4);
        // change byte-order
        num = (char*)&size;
        tmpByte = num[0];
        num[0] = num[3];
        num[3] = tmpByte;
        tmpByte = num[1];
        num[1] = num[2];
        num[2] = tmpByte;

        // skip the compressed block of data except for the last byte and
        // ignore the remaining bytes to prevent skipping over the end
        // of the file
        xtcFile.seekg(size - 1, std::ios_base::cur);
        xtcFile.ignore(((4 - size % 4) % 4) + 1);

        // add this frame to the frame count
        this->numXTCFrames++;
    }
    xtcFile.close();

    // remove the last frame
    this->XTCFrameOffset.RemoveLast();
    this->numXTCFrames--;

    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
        "Time for parsing the XTC-file: %f",
        (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG

    return true;
}

/*
 * Write all frames except for the first one from the currently loaded PDB-file
 * into a new XTC-file.
 */
void GROLoader::writeToXtcFile(const vislib::TString& filename) {

    std::ofstream outfile;
    unsigned int i;
    float precision = 1000.0;
    float minFloats[3];
    float maxFloats[3];

    if (data.Count() == 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "The PDB-file only contains one frame. No XTC-file has been"
            " written.");
        return;
    }

    // try to open the output-file
    outfile.open(filename, std::ios_base::binary | std::ios_base::out);

    // if the file could not be opened return
    if (!outfile) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Could not create file.");
        return;
    }

    // get the range of value from the existing bounding box
    minFloats[0] = this->bbox.Left(); // X-coord
    maxFloats[0] = this->bbox.Right();
    minFloats[1] = this->bbox.Bottom(); // Y-coord
    maxFloats[1] = this->bbox.Top();
    minFloats[2] = this->bbox.Back(); // Z-coord
    maxFloats[2] = this->bbox.Front();

    // loop through all frames
    for (i = 0; i < data.Count(); i++) {
        data[i]->writeFrame(&outfile, precision, minFloats, maxFloats);
    }

    // close the output-file
    outfile.close();
}
