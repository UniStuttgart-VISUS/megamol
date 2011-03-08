/*
 * GromacsLoader.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "GromacsLoader.h"
#include "param/FilePathParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/ASCIIFileBuffer.h"
#include <ctime>
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * GromacsLoader::Frame::Frame
 */
GromacsLoader::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner), atomCount( 0),
        maxBFactor(0), minBFactor( 0),
        maxCharge( 0), minCharge( 0),
        maxOccupancy( 0), minOccupancy( 0) {
    // Intentionally empty
}

/*
 * GromacsLoader::Frame::~Frame
 */
GromacsLoader::Frame::~Frame(void) {
}

/*
 * GromacsLoader::Frame::operator==
 */
bool GromacsLoader::Frame::operator==(const GromacsLoader::Frame& rhs) {
    // TODO: extend this accordingly
    return true;
}

/*
 * interpret a given bit array as an integer
 */
int GromacsLoader::Frame::decodebits(char *buff, int offset, int bitsize) {

    int num = 0;
    int mask = (1 << bitsize) -1; // '1'^bitsize

    // change byte order
    char tmpBuff[] = { buff[3], buff[2], buff[1], buff[0] };

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
void GromacsLoader::Frame::decodeints( char *buff, int offset, int num_of_bits,
                                   unsigned int sizes[], int nums[]) {


    int bytes[32];
    int i, j, num_of_bytes, p, num;

    bytes[1] = bytes[2] = bytes[3] = 0;
    num_of_bytes = 0;

    while (num_of_bits > 8)
    {
        // note: bit-offset stays the same
        bytes[num_of_bytes] = decodebits(buff, offset, 8);
        buff++;
        num_of_bytes++;
        num_of_bits -= 8;
    }
    if (num_of_bits > 0)
    {
        bytes[num_of_bytes++] = decodebits(buff, offset, num_of_bits);
    }

    // get num[2] and num[1]
    for (i = 2; i > 0; i--)
    {

        num = 0;
        for (j = num_of_bytes-1; j >=0; j--)
        {
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
unsigned int GromacsLoader::Frame::sizeofints(unsigned int sizes[]) {

    int i, num;
    unsigned int num_of_bytes, num_of_bits;
    unsigned int bytes[32], bytecnt, tmp;

    num_of_bytes = 1;
    bytes[0] = 1;
    num_of_bits = 0;

    for(i = 0; i < 3; i++)
    {
        tmp = 0;
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++)
        {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while (tmp != 0)
        {
            bytes[bytecnt++] = tmp & 0xff;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }

    num = 1;
    num_of_bytes--;
    while(bytes[num_of_bytes] >= (unsigned int)num)
    {
        num_of_bits++;
        num *= 2;
    }

    return num_of_bits + num_of_bytes * 8;

}

/*
 * sizeofint
 */
int GromacsLoader::Frame::sizeofint( int size ) {
    unsigned int num = 1;
    int num_of_bits = 0;

    while((unsigned int)size >= num && num_of_bits < 32)
    {
        num_of_bits++;
        num <<= 1;
    }
    return num_of_bits;
}

/*
 * change byte-order
 */
void GromacsLoader::Frame::changeByteOrder(char* num) {

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
void GromacsLoader::Frame::setFrameIdx(int idx) {
    this->frame = idx;
}

/*
 * encode the frame and write it to outfile
 */
 // TODO: handle the usage of large numbers
 // TODO: no compression for three atoms or less
bool GromacsLoader::Frame::writeFrame(std::ofstream *outfile, float precision,
                                  float *minFloats, float *maxFloats) {

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
    float *box = new float[9];
    box[0] = (float)sizes[0] / precision; box[1] = 0.0; box[2] = 0.0;
    box[3] = 0.0; box[4] = (float)sizes[1] / precision; box[5] = 0.0;
    box[6] = 0.0; box[7] = 0.0; box[8] = (float)sizes[2]/ precision;

    for(i = 0; i < 9; i++)
        changeByteOrder((char*)&box[i]);

    outfile->write((char*)box, 36);

    // write number of atoms to outfile
    outfile->write((char*)&atomCount, 4);

    // write precision to outfile
    float prec = precision*10.0f;
    changeByteOrder((char*)&prec);
    outfile->write((char*)&prec, 4);

    // write maxint[] and minint[] to outfile
    int maxVal[] = {maxInt[0], maxInt[1], maxInt[2]};
    int minVal[] = {minInt[0], minInt[1], minInt[2]};
    for(i = 0; i < 3; i++ ) {
        changeByteOrder((char*)&maxVal[i]);
        changeByteOrder((char*)&minVal[i]);
    }
    outfile->write((char*)&minVal, 12);
    outfile->write((char*)&maxVal, 12);

    // write smallidx to outfile
    int smallidx = 0;
    outfile->write((char*)&smallidx, 4);

    unsigned int bitoffset = 0;

    byteSize = ((bitsize+1) * (AtomCount() + 1)) / 8 + 1;
    
    // byteSize = actual size of the datablock
    // leave the rest filled with zeros
    char *charbuff = new char[byteSize + (4 -  byteSize % 4) % 4]; 
    char *charPt = charbuff;

    // important for bit-operations to work properly
    memset(charbuff, 0x00,  byteSize + ((4 -  byteSize % 4) % 4));

    // loop through all coordinate-triplets,transform coords to
    // unsigned ints and encode
    for(i = 0; i < AtomCount(); i++) {

        thiscoord[0] = (int)(atomPosition[i*3+0]*precision) - minInt[0];
        thiscoord[1] = (int)(atomPosition[i*3+1]*precision) - minInt[1];
        thiscoord[2] = (int)(atomPosition[i*3+2]*precision) - minInt[2];

        encodeints(charPt, bitsize, sizes, thiscoord, bitoffset);

        // update charPt
        charPt += ((bitoffset+bitsize) / 8);
        // calc new bitoffset
        bitoffset = (bitoffset+bitsize) % 8;


        // flag that runlength didn't change
        encodebits(charPt, 1, bitoffset, 0);

        // update charPt
        charPt += ((bitoffset+1) / 8);
        // calc new bitoffset
        bitoffset = (bitoffset+1) % 8;
    }

    // write the size to outfile
    unsigned int s =  byteSize;
    changeByteOrder((char*)&s);
    outfile->write((char*)&s, 4);

    // write buffer to file
    outfile->write(charbuff,  byteSize+((4 - byteSize % 4) % 4));

    delete[] box;
    delete[] charbuff;

    return true;
}

/*
 * encode ints
 *
 */
bool GromacsLoader::Frame::encodeints(char *outbuff, int num_of_bits,
                                  unsigned int sizes[], int inbuff[],
                                  unsigned int bitoffset) {

    int i;
    unsigned int bytes[32], num_of_bytes, bytecnt, tmp;

    tmp = inbuff[0];
    num_of_bytes = 0;
    char *buffPt = outbuff;

    // interpret every byte of the three ints as an unsigned int
    do {
        bytes[num_of_bytes++] = tmp & 0xff;
        tmp >>= 8;
    } while (tmp != 0);


    // loop trough all three ints and encode
    for(i = 1; i < 3; i++) {
        if(inbuff[i] >= sizes[i]) {
            return false;
        }

        // use one step multiply
        tmp = inbuff[i];
        for(bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while(tmp != 0) {
            bytes[bytecnt++] = tmp & 0xff;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }

    // write the result in the outbuffer
    if (num_of_bits >= num_of_bytes * 8) {
        for (i = 0; i < num_of_bytes; i++) {
            // bitsize = 8 --> offset doesn't change
            encodebits(buffPt, 8, bitoffset, bytes[i]);
            buffPt++;
        }
        encodebits(buffPt, num_of_bits - num_of_bytes * 8, bitoffset, 0);
    }
    else {
        for (i = 0; i < num_of_bytes-1; i++) {
            // bitsize = 8 --> offset doesn't change
            encodebits(buffPt, 8, bitoffset, bytes[i]);
            buffPt++;
        }
        encodebits(buffPt, num_of_bits- (num_of_bytes -1) * 8, bitoffset,
                   bytes[i]);
    }

    return true;
}

/*
 * encode an integer to binary
 */
void GromacsLoader::Frame::encodebits(char *outbuff, int bitsize, int bitoffset,
                                  unsigned int num ) {

    num <<= (32 - (bitoffset + bitsize));
    char *numpt = (char*)&num;

    // change byte order on little endian systems
    outbuff[0] = outbuff[0] | numpt[3];
    outbuff[1] = outbuff[1] | numpt[2];
    outbuff[2] = outbuff[2] | numpt[1];
    outbuff[3] = outbuff[3] | numpt[0];
}

/*
 * read frame-data from a given xtc-file
 */
void GromacsLoader::Frame::readFrame(std::fstream *file) {

    int *buffer;
    char *buffPt;
    int thiscoord[3],prevcoord[3],tempCoord;
    int run=0;
    unsigned int i=0;
    int bit_offset=0;

    unsigned int sizeint[3],sizesmall[3],bitsizeint[3];
    int flag;
    int smallnum,smaller,larger,is_smaller;
    unsigned int bitsize;
    unsigned int size;

    int minint[3],maxint[3];
    int smallidx;
    float precision;

    // note that magicints[FIRSTIDX-1] == 0
    const int magicints[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
        80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
        1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
        16384, 20642, 26007, 32768, 41285, 52015, 65536,82570, 104031,
        131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
        832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
        4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216
    };

    const int FIRSTIDX = 9;
    const int LASTIDX  = (sizeof(magicints) / sizeof(*magicints));


    // skip header data:
    // + version number     ( 4 Bytes)
    // + number of atoms    ( 4 Bytes)
    // + simulation step    ( 4 Bytes)
    // + simulation time    ( 4 Bytes)
    // + bounding box       (36 Bytes)
    // + number of atoms    ( 4 Bytes)
    file->seekg(56, std::ios_base::cur);


    // no compression is used for three atoms or less
    if(atomCount <= 3) {
        float posX, posY, posZ;
        for(i=0; i<atomCount; i++) {
            file->read((char*)&posX,4);
            changeByteOrder((char*)&posX);
            file->read((char*)&posY,4);
            changeByteOrder((char*)&posY);
            file->read((char*)&posZ,4);
            changeByteOrder((char*)&posZ);
            this->SetAtomPosition(i, posX, posY, posZ);
        }
        return;
    }

    // read the precision of the float coordinates
    file->read((char*)&precision,4);
    changeByteOrder((char*)&precision);
    precision /= 10.0f;

    // read the lower bound of 'big' integer-coordinates
    file->read((char*)&minint,12);
    changeByteOrder((char*)&minint[0]);
    changeByteOrder((char*)&minint[1]);
    changeByteOrder((char*)&minint[2]);

    // read the upper bound of 'big' integer-coordinates
    file->read((char*)&maxint,12);
    changeByteOrder((char*)&maxint[0]);
    changeByteOrder((char*)&maxint[1]);
    changeByteOrder((char*)&maxint[2]);


    sizeint[0] = maxint[0] - minint[0] + 1;
    sizeint[1] = maxint[1] - minint[1] + 1;
    sizeint[2] = maxint[2] - minint[2] + 1;

    // check if one of the sizes is to big to be multiplied
    if((sizeint[0] | sizeint[1] | sizeint[2] ) > 0xffffff)
    {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        bitsize = 0; // flag the use of large sizes
    }
    else
    {
        bitsizeint[0] = 0;
        bitsizeint[1] = 0;
        bitsizeint[2] = 0;
        bitsize = sizeofints(sizeint);
    }

    // read number of bits used to encode 'small' integers
    // note: changes dynamically within one frame
    file->read( (char*)&smallidx, 4 );
    changeByteOrder( (char*)&smallidx );

    // calculate maxidx/minidx
    int minidx, maxidx;
    if(LASTIDX < (unsigned int)(smallidx + 8)) {
        maxidx = LASTIDX;
    }
    else {
        maxidx = smallidx + 8;
    }
    minidx = maxidx - 8; // often this equal smallidx

    // if the difference to the last coordinate is smaller than smallnum
    // the difference is stored instead of the real coordinate
    smallnum = magicints[smallidx] / 2;

    // range of the 'small' integers
    sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx] ;

    // calculate smaller/larger
    if(FIRSTIDX>smallidx-1) {
        smaller = magicints[FIRSTIDX] / 2;
    }
    else {
        smaller = magicints[smallidx - 1] / 2;
    }
    larger = magicints[maxidx];

    // read the size of the compressed data-block
    file->read((char*)&size,4);
    changeByteOrder((char*)&size);

    buffer = new int[(int)(atomCount*3*1.2)];

    // get the compressed data-block
    file->read((char*)&buffer[0], size);

    buffPt = (char*)buffer;
    bit_offset = 0;


    while(i < atomCount) {

        thiscoord[0] = 0;
        thiscoord[1] = 0;
        thiscoord[2] = 0;


        // if large numbers are used
        if(bitsize == 0) {
            thiscoord[0] = decodebits(buffPt, bit_offset, bitsizeint[0]);
            buffPt += (bit_offset + bitsizeint[0]) / 8;
            bit_offset = (bit_offset + bitsizeint[0]) % 8;

            thiscoord[1] = decodebits(buffPt, bit_offset, bitsizeint[1]);
            buffPt += (bit_offset + bitsizeint[1]) / 8;
            bit_offset = (bit_offset + bitsizeint[1]) % 8;

            thiscoord[2] = decodebits(buffPt, bit_offset, bitsizeint[2]);
            buffPt += (bit_offset + bitsizeint[2]) / 8;
            bit_offset = (bit_offset + bitsizeint[2]) % 8;
        }
        else {
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
        if(flag == 1) {
            run = decodebits(buffPt, bit_offset, 5);
            buffPt += (bit_offset + 5) / 8;
            bit_offset = (bit_offset + 5) % 8;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller--;
        }

        // run = the number of coordinates following the current coordinate that
        // have bin stored as differences to there previous coordinate
        if(run > 0)
        {
            // save the current coordinate
            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];

            for(int k = 0; k < run; k+=3)
            {
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
                    this->SetAtomPosition(i, (float)prevcoord[0] / precision,
                                          (float)prevcoord[1] / precision,
                                          (float)prevcoord[2] / precision);
                    i++;
                }
                else {
                    prevcoord[0] = thiscoord[0];
                    prevcoord[1] = thiscoord[1];
                    prevcoord[2] = thiscoord[2];
                }

                this->SetAtomPosition(i, (float)thiscoord[0] / precision,
                                      (float)thiscoord[1] / precision,
                                      (float)thiscoord[2] / precision);
                i++;
            }
        } else {
            this->SetAtomPosition(i, (float)thiscoord[0] / precision,
                                     (float)thiscoord[1] / precision,
                                     (float)thiscoord[2] / precision);
            i++;
        }


        // update smallidx etc
        smallidx += is_smaller;
        if(is_smaller < 0) {
            smallnum = smaller;
            if (smallidx > FIRSTIDX) {
                smaller = magicints[smallidx - 1] /2;
            }
            else {
                smaller = 0;
            }
        }
        else if(is_smaller > 0) {
            smaller = smallnum;
            smallnum = magicints[smallidx] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx] ;
    }

    delete[] buffer;

    // set file pointer to the beginning of the next frame
    file->seekg((4 - size % 4) % 4, std::ios_base::cur);
}

/*
 * Assign a position to the array of positions.
 */
bool GromacsLoader::Frame::SetAtomPosition( unsigned int idx, float x, float y,
                                        float z) {
    if( idx >= this->atomCount ) return false;
    this->atomPosition[idx*3+0] = x;
    this->atomPosition[idx*3+1] = y;
    this->atomPosition[idx*3+2] = z;
    return true;
}

/*
 * Assign a position to the array of positions.
 */
bool GromacsLoader::Frame::SetAtomBFactor( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->bfactor[idx] = val;
    return true;
}

/*
 * Assign a charge to the array of charges.
 */
bool GromacsLoader::Frame::SetAtomCharge( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->charge[idx] = val;
    return true;
}

/*
 * Assign a occupancy to the array of occupancies.
 */
bool GromacsLoader::Frame::SetAtomOccupancy( unsigned int idx, float val) {
    if( idx >= this->atomCount ) return false;
    this->occupancy[idx] = val;
    return true;
}

// ======================================================================

/*
 * protein::GromacsLoader::GromacsLoader
 */
GromacsLoader::GromacsLoader(void) : AnimDataModule(),
        topFilenameSlot( "topologyFile", "The path to the topology data file to be loaded"),
        xtcFilenameSlot( "trajectoryFile", "The path to the trajectory data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        strideFlagSlot( "strideFlag", "The flag wether STRIDE should be used or not."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0),
        stride( 0), secStructAvailable( false), numXTCFrames( 0),
        XTCFrameOffset( 0), xtcFileValid(false),
        tpx_incompatible_version( 9), tpx_version( 73), tpx_generation( 23) {

    this->topFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->topFilenameSlot);

    this->xtcFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->xtcFilenameSlot);

    this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &GromacsLoader::getData);
    this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &GromacsLoader::getExtent);
    this->MakeSlotAvailable( &this->dataOutSlot);

    this->strideFlagSlot << new param::BoolParam(true);
    this->MakeSlotAvailable( &this->strideFlagSlot);

}

/*
 * protein::GromacsLoader::~GromacsLoader
 */
GromacsLoader::~GromacsLoader(void) {
    this->Release ();
}

/*
 * GromacsLoader::create
 */
bool GromacsLoader::create(void) {
    // intentionally empty
    return true;
}

/*
 * GromacsLoader::getData
 */
bool GromacsLoader::getData( core::Call& call) {
    using vislib::sys::Log;

    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->topFilenameSlot.IsDirty() ) {
        this->topFilenameSlot.ResetDirty();
        this->loadFile( this->topFilenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    dc->SetDataHash( this->datahash);

    // if no xtc-filename has been set
    if( !this->xtcFileValid) {

        //if( dc->FrameID() >= this->data.Count() ) return false;
        if( dc->FrameID() >= this->data.Count() ) {
            if( this->data.Count() > 0 ) {
                dc->SetFrameID( this->data.Count() -  1);
            } else {
                return false;
            }
        }

        dc->SetAtoms(this->data[dc->FrameID()]->AtomCount(),
             this->atomType.Count(),
             (unsigned int*)this->atomTypeIdx.PeekElements(),
             (float*)this->data[dc->FrameID()]->AtomPositions(),
             (MolecularDataCall::AtomType*)this->atomType.PeekElements(),
             (float*)this->data[dc->FrameID()]->AtomBFactor(),
             (float*)this->data[dc->FrameID()]->AtomCharge(),
             (float*)this->data[dc->FrameID()]->AtomOccupancy());

        dc->SetBFactorRange( this->data[dc->FrameID()]->MinBFactor(),
            this->data[dc->FrameID()]->MaxBFactor());
        dc->SetChargeRange( this->data[dc->FrameID()]->MinCharge(),
            this->data[dc->FrameID()]->MaxCharge());
        dc->SetOccupancyRange( this->data[dc->FrameID()]->MinOccupancy(),
            this->data[dc->FrameID()]->MaxOccupancy());
    } else {

        Frame *fr = NULL;
        fr = dynamic_cast<GromacsLoader::Frame *>(this->
               requestLockedFrame(dc->FrameID()));

        if (fr == NULL)
            return false;

        dc->SetUnlocker(new Unlocker(*fr));

        dc->SetAtoms( this->data[0]->AtomCount(),
                      this->atomType.Count(),
                      (unsigned int*)this->atomTypeIdx.PeekElements(),
                      (float*)fr->AtomPositions(),
                      (MolecularDataCall::AtomType*)this->atomType.PeekElements(),
                      (float*)this->data[0]->AtomBFactor(),
                      (float*)this->data[0]->AtomCharge(),
                      (float*)this->data[0]->AtomOccupancy());

        dc->SetBFactorRange( this->data[0]->MinBFactor(),
                             this->data[0]->MaxBFactor());
        dc->SetChargeRange( this->data[0]->MinCharge(),
                            this->data[0]->MaxCharge());
        dc->SetOccupancyRange( this->data[0]->MinOccupancy(),
                               this->data[0]->MaxOccupancy());
    }

    dc->SetConnections( this->connectivity.Count() / 2,
        (const unsigned int*)this->connectivity.PeekElements());
    dc->SetResidues( this->residue.Count(),
        (const MolecularDataCall::Residue**)this->residue.PeekElements());
    dc->SetResidueTypeNames( this->residueTypeName.Count(),
        (const vislib::StringA*)this->residueTypeName.PeekElements());
    dc->SetMolecules( this->molecule.Count(),
        (const MolecularDataCall::Molecule*)this->molecule.PeekElements());
    dc->SetChains( this->chain.Count(),
        (const MolecularDataCall::Chain*)this->chain.PeekElements());

    if( !this->secStructAvailable &&
            this->strideFlagSlot.Param<param::BoolParam>()->Value() ) {
        time_t t = clock(); // DEBUG
        if( this->stride ) delete this->stride;
        this->stride = new Stride( dc);
        this->stride->WriteToInterface( dc);
        this->secStructAvailable = true;
        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Secondary Structure computed via STRIDE in %f seconds.", ( double( clock() - t) / double( CLOCKS_PER_SEC))); // DEBUG
    } else if( this->strideFlagSlot.Param<param::BoolParam>()->Value() ) {
        this->stride->WriteToInterface( dc);
    }

    return true;
}

/*
 * GromacsLoader::getExtent
 */
bool GromacsLoader::getExtent( core::Call& call) {
    MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->topFilenameSlot.IsDirty() ) {
        this->topFilenameSlot.ResetDirty();
        this->loadFile( this->topFilenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    // grow bounding box by 3.0 Angstrom (for volume rendering / SAS)
    vislib::math::Cuboid<float> bBoxPlus3( this->bbox);
    bBoxPlus3.Grow( 3.0f);

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox( bBoxPlus3);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox( bBoxPlus3);

    if( !xtcFileValid ) {
        // no XTC file set or loaded --> use number of loaded frames
        dc->SetFrameCount( vislib::math::Max(1U,
                           static_cast<unsigned int>( this->data.Count())));
    }
    else {
        // XTC file set and loaded --> use number of frames
        dc->SetFrameCount( vislib::math::Max(1U, static_cast<unsigned int>(
                           this->numXTCFrames)));
    }

    dc->SetDataHash( this->datahash);

    return true;
}

/*
 * GromacsLoader::release
 */
void GromacsLoader::release(void) {
    // stop frame-loading thread before clearing data array
    resetFrameCache();
    for(int i = 0; i < this->data.Count(); i++)
        delete data[i];
    this->data.Clear();
}


/*
 * GromacsLoader::constructFrame
 */
view::AnimDataModule::Frame* GromacsLoader::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<GromacsLoader*>(this));
    f->SetAtomCount( this->data[0]->AtomCount() );
    return f;
}

/*
 * GromacsLoader::loadFrame
 */
void GromacsLoader::loadFrame( view::AnimDataModule::Frame *frame,
                           unsigned int idx) {

    //time_t t = clock();

    GromacsLoader::Frame *fr = dynamic_cast<GromacsLoader::Frame*>(frame);
    //int atomCnt = this->data[0]->AtomCount();

    // set the frames index
    fr->setFrameIdx( idx);

    // read first frame from the existing data-buffer
    /*if( idx == 0 ) {
        for( unsigned int i = 0; i < atomCnt*3; i+=3 ) {
            fr->SetAtomPosition(i/3, data[0]->AtomPositions()[i],
                                data[0]->AtomPositions()[i+1],
                                data[0]->AtomPositions()[i+2]);
        }
    } else {*/
        std::fstream xtcFile;

        xtcFile.open(this->xtcFilenameSlot.
          Param<core::param::FilePathParam>()->Value(),
          std::ios::in | std::ios::binary);

        xtcFile.seekg( this->XTCFrameOffset[idx]);

        fr->readFrame(&xtcFile);

        xtcFile.close();
    //}

    //vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
    //"Time for loading frame %i: %f", idx,
    //( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG
}

/*
 * GromacsLoader::loadFile
 */
void GromacsLoader::loadFile( const vislib::TString& filename) {
    using vislib::sys::Log;

    this->resetAllData();

    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    for(int i = 0; i < this->data.Count(); i++)
        delete data[i];
    this->data.Clear();

    // stop frame-loading thread if neccessary
    if( this->xtcFileValid )
        resetFrameCache();

    this->data.Clear();
    this->datahash++;


    vislib::StringA line;
    unsigned int idx, atomCnt, lineCnt, frameCnt, resCnt, chainCnt;

    time_t t = clock(); // DEBUG

    // try to open the specified TPR file
    FILE *bf = fopen( T2A(filename), "rb");
    // if the file was successfully opened...
    if( bf ) {
        // create a new XDR data object
        XDR *xdr = new XDR();
        xdrstdio_create( xdr, bf, /*xdr_op::*/XDR_DECODE);

        // read TPX header from the XDR file
        if( !this->readTpxHeader( xdr, this->tpx) ) {
            Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
                "Could not load header of file %s.", 
                static_cast<const char*>(T2A( filename)));
            return;
        }

        // read the state from the XDR file
        this->readState( xdr, this->state, tpx.natoms, tpx.ngtc);

        // close the file
        fclose( bf);
        // destroy the XDR data object
        xdr_destroy( xdr);

        Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for loading file %s: %f", 
            static_cast<const char*>(T2A( filename)), 
            ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

        // xtc-filename has been set
        if( !this->xtcFilenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty() ) {
            // try to get the total number of frames and calculate the
            // bounding box
            this->readNumXTCFrames();

            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO,
                "Number of XTC-frames: %u", this->numXTCFrames); // DEBUG

            //float box[3][3];
            char tmpByte;
            std::fstream xtcFile;
            char *num;
            unsigned int nAtoms;

            // try to open the xtc-file
            xtcFile.open(this->xtcFilenameSlot.
                           Param<core::param::FilePathParam>()->Value(),
                           std::ios::in | std::ios::binary);

            if( !xtcFile) {
                Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
                  "Could not load XTC-file."); // DEBUG
                xtcFileValid = false;
            } else {

                xtcFile.seekg(4, std::ios_base::cur);
                // read number of atoms
                xtcFile.read((char*)&nAtoms, 4);
                // change byte order
                num = (char*)&nAtoms;
                tmpByte = num[0]; num[0] = num[3]; num[3] = tmpByte;
                tmpByte = num[1]; num[1] = num[2]; num[2] = tmpByte;

                // check whether the pdb-file and the xtc-file contain the
                // same number of atoms
                if( nAtoms != this->tpx.natoms ) {
                    Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
                      "XTC-File and given PDB-file not matching (XTC-file has"
                      "%i atom entries, PDB-file has %i atom entries).",
                         nAtoms, this->tpx.natoms); // DEBUG
                    xtcFileValid = false;
                    xtcFile.close();
                }
                else {
                    xtcFile.close();

                    xtcFileValid = true;

                    int maxFrames = this->numXTCFrames;

                    // frames in xtc-file - 1 (without the last frame)
                    this->setFrameCount( this->numXTCFrames);

                    // start the loading thread
                    this->initFrameCache( maxFrames);
                }
            }
        }

    } else {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
            "Could not open file %s", 
            static_cast<const char*>(T2A( filename)));
    }
}

/*
 * Get a real number from a xdr file
 */
float GromacsLoader::getXdrReal( XDR *xdr) {
    double d;
    float f;
    if( xdr->bDouble ) {
        xdr_double( xdr, &d);
        f = float( d);
    } else {
        xdr_float( xdr, &f);
    }
    return f;
}

/*
 * Read the tpx header of a XDR file
 */
bool GromacsLoader::readTpxHeader( XDR *xdr, TpxHeader &tpx) {
    using vislib::sys::Log;
    // temporary variables
    int ssize( 0);
    char *strg = 0;
    int fver, fgen, idum;
    float rdum;

    xdr_int( xdr, &ssize);
    strg = new char[ssize];
    xdr_string( xdr, &strg, ssize);
    if( strncmp( strg, "VERSION", 7) ) {
      Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
          "Can not read file header, this file is from a Gromacs version which is older than 2.0");
    }
    if( strg ) delete[] strg;

    int precision = 0;
    xdr_int( xdr, &precision);

    xdr->bDouble = (precision == sizeof(double));
    if( (precision != sizeof(float)) && !xdr->bDouble) { 
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
            "Unknown precision: real is %d bytes instead of %d or %d", 
            precision, sizeof(float), sizeof(double));
        return false;
    }

    // Check versions!
    xdr_int( xdr, &fver);

    if( fver >= 26 )
        xdr_int( xdr, &fgen);
    else
        fgen=0;

    this->fileVersion = fver;
    this->fileGeneration = fgen;

    if ((fver <= tpx_incompatible_version) || (fgen > tpx_generation)) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
            "Reading TPR file version %d with version %d program", 
            fver, tpx_version);
        return false;
    }
    
    //do_section(fio,eitemHEADER,bRead);

    xdr_int( xdr, &tpx.natoms);
    if( fver >= 28 ) {
        xdr_int( xdr, &tpx.ngtc);
    } else {
        tpx.ngtc = 0;
    }
    if( fver < 62 ) {
        xdr_int( xdr, &idum);
        rdum = this->getXdrReal( xdr);
    }
    
    tpx.lambda = this->getXdrReal( xdr);
    xdr_int( xdr, &tpx.bIr);
    xdr_int( xdr, &tpx.bTop);
    xdr_int( xdr, &tpx.bX);
    xdr_int( xdr, &tpx.bV);
    xdr_int( xdr, &tpx.bF);
    xdr_int( xdr, &tpx.bBox);

    if( fgen > tpx_generation ) {
        // This can only happen if TopOnlyOK=TRUE
        tpx.bIr = FALSE;
    }
    
    return true;
}

/*
 * Read the state from a XDR file
 */
bool GromacsLoader::readState( XDR *xdr, t_state &state, int natoms, int ngtc) {
    using vislib::sys::Log;
    int i;
    gmx_bool bDum = TRUE;

    // ---------- init state ----------
    state.natoms = natoms;
    state.nrng   = 0;
    state.flags  = 0;
    state.lambda = 0;
    state.veta   = 0;
    clear_mat( state.box);
    clear_mat( state.box_rel);
    clear_mat( state.boxv);
    clear_mat( state.pres_prev);
    clear_mat( state.svir_prev);
    clear_mat( state.fvir_prev);
    init_gtc_state( state, ngtc, 0, 0);
    state.nalloc = state.natoms;
    if (state.nalloc > 0) {
        state.x = new rvec[state.nalloc];
        state.v = new rvec[state.nalloc];
    } else {
        state.x = NULL;
        state.v = NULL;
    }
    state.sd_X = NULL;
    state.cg_p = NULL;

    init_ekinstate( &state.ekinstate);

    init_energyhistory( &state.enerhist);

    state.ddp_count = 0;
    state.ddp_count_cg_gl = 0;
    state.cg_gl = NULL;
    state.cg_gl_nalloc = 0;

    // ---------- read state ----------
    do_xdr( xdr, &state.box, DIM, eioNRVEC);
    if( this->fileVersion >= 51) {
        do_xdr( xdr, &state.box_rel, DIM, eioNRVEC);
    } else {
        // We initialize box_rel after reading the inputrec
        clear_mat( state.box_rel);
    }
    if( this->fileVersion >= 28 ) {
        do_xdr( xdr, &state.boxv, DIM, eioNRVEC);
        if( this->fileVersion < 56 ) {
            matrix mdum;
            do_xdr( xdr, &mdum, DIM, eioNRVEC);
        }
    }

    if( state.ngtc > 0 && this->fileVersion >= 28) {
        real *dumv;
        dumv = new real[state.ngtc];
        if( this->fileVersion < 69) {
            bDum = do_xdr( xdr, dumv, state.ngtc, eioREAL);
        }
        /* These used to be the Berendsen tcoupl_lambda's */
        bDum = do_xdr( xdr, dumv, state.ngtc, eioREAL);
        delete[] dumv;
    }

    // Prior to tpx version 26, the inputrec was here.
    // I moved it to enable partial forward-compatibility
    // for analysis/viewer programs.
    if( this->fileVersion < 26 ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
            "Can not load file versions lower than 26.");
        /*
        do_test(fio,tpx.bIr,ir);
        do_section(fio,eitemIR,bRead);
        if (tpx.bIr) {
            if (ir) {
                do_inputrec(fio, ir,bRead,file_version,
                mtop ? &mtop->ffparams.fudgeQQ : NULL);
                if (bRead && debug) {
                    pr_inputrec(debug,0,"inputrec",ir,FALSE);
                }
            } else {
                do_inputrec(fio, &dum_ir,bRead,file_version,
                    mtop ? &mtop->ffparams.fudgeQQ :NULL);
                if (bRead && debug) {
                    pr_inputrec(debug,0,"inputrec",&dum_ir,FALSE);
                }
                done_inputrec(&dum_ir);
            }
        }
        */

    }

	/*
    if( tpx.bTop ) {
        if( mtop ) {
            do_mtop(fio,mtop,bRead, file_version);
        } else {
            do_mtop(fio,&dum_top,bRead,file_version);
            done_mtop(&dum_top,TRUE);
        }
    }
    if( tpx.bX ) {
        if( bRead ) {
            state->flags |= (1<<estX);
        }
        gmx_fio_ndo_rvec(fio,state->x,state->natoms);
    }
	*/

    return true;
}

/*
 * initialize gtc_state
 */
void GromacsLoader::init_gtc_state(t_state &state, int ngtc, int nnhpres, int nhchainlength) {
    int i,j;

    state.ngtc = ngtc;
    state.nnhpres = nnhpres;
    state.nhchainlength = nhchainlength;
    if (state.ngtc > 0) {
        state.nosehoover_xi = new double[state.nhchainlength*state.ngtc]; 
        state.nosehoover_vxi = new double[state.nhchainlength*state.ngtc];
        state.therm_integral = new double[state.ngtc];
        for(i=0; i<state.ngtc; i++) {
            for (j=0;j<state.nhchainlength;j++) {
                state.nosehoover_xi[i*state.nhchainlength + j]  = 0.0;
                state.nosehoover_vxi[i*state.nhchainlength + j]  = 0.0;
            }
        }
        for(i=0; i<state.ngtc; i++) {
            state.therm_integral[i]  = 0.0;
        }
    } else {
        state.nosehoover_xi  = NULL;
        state.nosehoover_vxi = NULL;
        state.therm_integral = NULL;
    }

    if (state.nnhpres > 0)
    {
        state.nhpres_xi = new double[state.nhchainlength*nnhpres];
        state.nhpres_vxi = new double[state.nhchainlength*nnhpres];
        for(i=0; i<nnhpres; i++) {
            for (j=0;j<state.nhchainlength;j++) {
                state.nhpres_xi[i*nhchainlength + j]  = 0.0;
                state.nhpres_vxi[i*nhchainlength + j]  = 0.0;
            }
        }
    } else {
        state.nhpres_xi  = NULL;
        state.nhpres_vxi = NULL;
    }
}

/*
 * initialize ekinstate
 */
void GromacsLoader::init_ekinstate(ekinstate_t *eks) {
  eks->ekin_n         = 0;
  eks->ekinh          = NULL;
  eks->ekinf          = NULL;
  eks->ekinh_old      = NULL;
  eks->ekinscalef_nhc = NULL;
  eks->ekinscaleh_nhc = NULL;
  eks->vscale_nhc     = NULL;
  eks->dekindl        = 0;
  eks->mvcos          = 0;
}

/*
 * initialize energyhistory
 */
void GromacsLoader::init_energyhistory(energyhistory_t *enerhist) {
    enerhist->nener = 0;

    enerhist->ener_ave     = NULL;
    enerhist->ener_sum     = NULL;
    enerhist->ener_sum_sim = NULL;
    enerhist->dht          = NULL;

    enerhist->nsteps     = 0;
    enerhist->nsum       = 0;
    enerhist->nsteps_sim = 0;
    enerhist->nsum_sim   = 0;

    enerhist->dht = NULL;
}

/*
 * do_xdr
 */
bool GromacsLoader::do_xdr( XDR *xdr, void *item, int nitem, int eio) {
    unsigned char ucdum, *ucptr;
    bool_t res = 0;
    float fvec[DIM];
    double dvec[DIM];
    int j, m, *iptr, idum;
    gmx_large_int_t sdum;
    real *ptr;
    unsigned short us;
    double d = 0;
    float f = 0;

    switch (eio)
    {
    case eioREAL:
        if (xdr->bDouble)
        {
            if (item )
                d = *((real *) item);
            res = xdr_double( xdr, &d);
            if (item)
                *((real *) item) = d;
        }
        else
        {
            if (item )
                f = *((real *) item);
            res = xdr_float(xdr, &f);
            if (item)
                *((real *) item) = f;
        }
        break;
    case eioFLOAT:
        if (item )
            f = *((float *) item);
        res = xdr_float(xdr, &f);
        if (item)
            *((float *) item) = f;
        break;
    case eioDOUBLE:
        if (item )
            d = *((double *) item);
        res = xdr_double(xdr, &d);
        if (item)
            *((double *) item) = d;
        break;
    case eioINT:
        if (item )
            idum = *(int *) item;
        res = xdr_int(xdr, &idum);
        if (item)
            *(int *) item = idum;
        break;
    /*
    case eioGMX_LARGE_INT:
        // do_xdr will not generate a warning when a 64bit gmx_large_int_t
        // value that is out of 32bit range is read into a 32bit gmx_large_int_t.
        if (item )
            sdum = *(gmx_large_int_t *) item;
        res = xdr_gmx_large_int(xdr, &sdum, NULL);
        if (item)
            *(gmx_large_int_t *) item = sdum;
        break;
    */
    case eioUCHAR:
        if (item )
            ucdum = *(unsigned char *) item;
        res = xdr_u_char(xdr, &ucdum);
        if (item)
            *(unsigned char *) item = ucdum;
        break;
    case eioNUCHAR:
        ucptr = (unsigned char *) item;
        res = 1;
        for (j = 0; (j < nitem) && res; j++)
        {
            res = xdr_u_char(xdr, &(ucptr[j]));
        }
        break;
    case eioUSHORT:
        if (item )
            us = *(unsigned short *) item;
        res = xdr_u_short(xdr, (unsigned short *) &us);
        if (item)
            *(unsigned short *) item = us;
        break;
    case eioRVEC:
        if (xdr->bDouble)
        {
            if (item )
                for (m = 0; (m < DIM); m++)
                    dvec[m] = ((real *) item)[m];
            res = xdr_vector(xdr, (char *) dvec, DIM,
                             (unsigned int) sizeof(double),
                             (xdrproc_t) xdr_double);
            if (item)
                for (m = 0; (m < DIM); m++)
                    ((real *) item)[m] = dvec[m];
        }
        else
        {
            if (item )
                for (m = 0; (m < DIM); m++)
                    fvec[m] = ((real *) item)[m];
            res = xdr_vector(xdr, (char *) fvec, DIM,
                             (unsigned int) sizeof(float),
                             (xdrproc_t) xdr_float);
            if (item)
                for (m = 0; (m < DIM); m++)
                    ((real *) item)[m] = fvec[m];
        }
        break;
    case eioNRVEC:
        ptr = NULL;
        res = 1;
        for (j = 0; (j < nitem) && res; j++)
        {
            if (item)
                ptr = ((rvec *) item)[j];
            res = do_xdr( xdr, ptr, 1, eioRVEC);
        }
        break;
    case eioIVEC:
        iptr = (int *) item;
        res = 1;
        for (m = 0; (m < DIM) && res; m++)
        {
            if (item )
                idum = iptr[m];
            res = xdr_int(xdr, &idum);
            if (item)
                iptr[m] = idum;
        }
        break;
    case eioSTRING:
    {
        char *cptr = 0;
        int slen;

        slen = 0;
            
        if (xdr_int(xdr, &slen) <= 0)
            printf( "wrong string length %d for string",slen);
        if (!item )
            cptr = new char[slen];
        else
            cptr=(char *)item;
        if (cptr)
            res = xdr_string(xdr,&cptr,slen);
        else
            res = 1;
        if (!item )
            delete[] cptr;
        break;
    }
    }

    return (res != 0);
}


/*
 * Get the radius of the element
 */
float GromacsLoader::getElementRadius( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    // --- van der Waals radii ---
    if( name[cnt] == 'H' )
        return 1.2f;
    if( name[cnt] == 'C' )
        return 1.7f;
    if( name[cnt] == 'N' )
        return 1.55f;
    if( name[cnt] == 'O' )
        return 1.52f;
    if( name[cnt] == 'S' )
        return 1.8f;
    if( name[cnt] == 'P' )
        return 1.8f;
    if( name[cnt] == 'C' )
        return 1.7f;

    return 1.5f;
}

/*
 * Get the color of the element
 */
vislib::math::Vector<unsigned char, 3> GromacsLoader::getElementColor( vislib::StringA name) {
    // extract the element symbol from the name
    unsigned int cnt = 0;
    vislib::StringA element;
    while( vislib::CharTraitsA::IsDigit( name[cnt]) ) {
        cnt++;
    }

    if( name[cnt] == 'H' ) // white or light grey
        return vislib::math::Vector<unsigned char, 3>( 240, 240, 240);
    if( name[cnt] == 'C' ) // (dark) grey or green
        return vislib::math::Vector<unsigned char, 3>( 125, 125, 125);
        //return vislib::math::Vector<unsigned char, 3>( 90, 175, 50);
    if( name[cnt] == 'N' ) // blue
        //return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
        return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
    if( name[cnt] == 'O' ) // red
        //return vislib::math::Vector<unsigned char, 3>( 250, 94, 82);
        return vislib::math::Vector<unsigned char, 3>( 206, 34, 34);
    if( name[cnt] == 'S' ) // yellow
        //return vislib::math::Vector<unsigned char, 3>( 250, 230, 50);
        return vislib::math::Vector<unsigned char, 3>( 255, 215, 0);
    if( name[cnt] == 'P' ) // orange
        return vislib::math::Vector<unsigned char, 3>( 255, 128, 64);

    return vislib::math::Vector<unsigned char, 3>( 191, 191, 191);
}


/*
 * Search for connections in the given residue and add them to the
 * global connection array.
 */
void GromacsLoader::MakeResidueConnections( unsigned int resIdx, unsigned int frame) {
    // check bounds
    if( resIdx >= this->residue.Count() ) return;
    if( frame >= this->data.Count() ) return;
    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if( this->connectivity.Count() == this->connectivity.Capacity() ) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity( connectionCapacity);
    }
    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for( cnt0 = 0; cnt0 < this->residue[resIdx]->AtomCount() - 1; ++cnt0 ) {
        for( cnt1 = cnt0 + 1; cnt1 < this->residue[resIdx]->AtomCount(); ++cnt1 ) {
            // get atom indices
            atomIdx0 = this->residue[resIdx]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set( this->data[frame]->AtomPositions()[3*atomIdx0+0],
                this->data[frame]->AtomPositions()[3*atomIdx0+1],
                this->data[frame]->AtomPositions()[3*atomIdx0+2]);
            atomPos1.Set( this->data[frame]->AtomPositions()[3*atomIdx1+0],
                this->data[frame]->AtomPositions()[3*atomIdx1+1],
                this->data[frame]->AtomPositions()[3*atomIdx1+2]);
            // check distance
            if( ( atomPos0 - atomPos1).Length() <
                0.58f * ( this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                this->atomType[this->atomTypeIdx[atomIdx1]].Radius() ) ) {
                // add connection
                this->connectivity.Add( atomIdx0);
                this->connectivity.Add( atomIdx1);
            }
        }
    }
}

/*
 * Search for connections between two residues.
 */
bool GromacsLoader::MakeResidueConnections( unsigned int resIdx0, unsigned int resIdx1, unsigned int frame) {
    // flag wether the two residues are connected
    bool connected = false;
    // check bounds
    if( resIdx0 >= this->residue.Count() ) return connected;
    if( resIdx1 >= this->residue.Count() ) return connected;
    if( frame >= this->data.Count() ) return connected;

    // get capacity of connectivity array
    SIZE_T connectionCapacity = this->connectivity.Capacity();
    // increase capacity of connectivity array, of necessary
    if( this->connectivity.Count() == this->connectivity.Capacity() ) {
        connectionCapacity += 10000;
        this->connectivity.AssertCapacity( connectionCapacity);
    }

    // loop over all atoms in the residue
    unsigned int cnt0, cnt1, atomIdx0, atomIdx1;
    vislib::math::Vector<float, 3> atomPos0, atomPos1;
    for( cnt0 = 0; cnt0 < this->residue[resIdx0]->AtomCount(); ++cnt0 ) {
        for( cnt1 = 0; cnt1 < this->residue[resIdx1]->AtomCount(); ++cnt1 ) {
            // get atom indices
            atomIdx0 = this->residue[resIdx0]->FirstAtomIndex() + cnt0;
            atomIdx1 = this->residue[resIdx1]->FirstAtomIndex() + cnt1;
            // get atom positions
            atomPos0.Set( this->data[frame]->AtomPositions()[3*atomIdx0+0],
                this->data[frame]->AtomPositions()[3*atomIdx0+1],
                this->data[frame]->AtomPositions()[3*atomIdx0+2]);
            atomPos1.Set( this->data[frame]->AtomPositions()[3*atomIdx1+0],
                this->data[frame]->AtomPositions()[3*atomIdx1+1],
                this->data[frame]->AtomPositions()[3*atomIdx1+2]);
            // check distance
            if( ( atomPos0 - atomPos1).Length() <
                0.58f * ( this->atomType[this->atomTypeIdx[atomIdx0]].Radius() +
                this->atomType[this->atomTypeIdx[atomIdx1]].Radius() ) ) {
                // add connection
                this->connectivity.Add( atomIdx0);
                this->connectivity.Add( atomIdx1);
                connected = true;
            }
        }
    }
    return connected;
}

/*
 * Check if the residue is an amino acid.
 */
bool GromacsLoader::IsAminoAcid( vislib::StringA resName ) {
    if( resName.Equals( "ALA" ) |
        resName.Equals( "ARG" ) |
        resName.Equals( "ASN" ) |
        resName.Equals( "ASP" ) |
        resName.Equals( "CYS" ) |
        resName.Equals( "GLN" ) |
        resName.Equals( "GLU" ) |
        resName.Equals( "GLY" ) |
        resName.Equals( "HIS" ) |
        resName.Equals( "ILE" ) |
        resName.Equals( "LEU" ) |
        resName.Equals( "LYS" ) |
        resName.Equals( "MET" ) |
        resName.Equals( "PHE" ) |
        resName.Equals( "PRO" ) |
        resName.Equals( "SER" ) |
        resName.Equals( "THR" ) |
        resName.Equals( "TRP" ) |
        resName.Equals( "TYR" ) |
        resName.Equals( "VAL" ) |
        resName.Equals( "ASH" ) |
        resName.Equals( "CYX" ) |
        resName.Equals( "CYM" ) |
        resName.Equals( "GLH" ) |
        resName.Equals( "HID" ) |
        resName.Equals( "HIE" ) |
        resName.Equals( "HIP" ) |
        resName.Equals( "LYN" ) |
        resName.Equals( "TYM" ) )
        return true;
    return false;
}

/*
 * reset all data containers.
 */
void GromacsLoader::resetAllData() {
    unsigned int cnt;
    //this->data.Clear();
    this->atomTypeIdx.Clear();
    this->atomType.Clear();
    for( cnt = 0; cnt < this->residue.Count(); ++cnt ) {
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
}


/*
 * Read the number of frames from the XTC file and update the bounding box.
 * The Last frame contains wrong byte ordering and therefore gets ignored.
 */
bool GromacsLoader::readNumXTCFrames() {

    time_t t = clock();

    // reset values
    this->numXTCFrames = 0;
    this->XTCFrameOffset.Clear();

    // try to open xtc file
    std::fstream xtcFile;
    xtcFile.open(this->xtcFilenameSlot.
      Param<core::param::FilePathParam>()->Value(),
      std::ios::in | std::ios::binary);

    // check if file could be opened
    if( !xtcFile ) return false;

    this->XTCFrameOffset.SetCapacityIncrement( 1000);
    int size;
    char tmpByte;
    char *num;

    int minint[3];
    int maxint[3];

    unsigned int i;

    float precision;

    xtcFile.seekg(0, std::ios_base::beg);

    vislib::math::Cuboid<float> tmpBBox( this->bbox);

    // read until eof
    while( !xtcFile.eof() ) {

        // add the offset to the offset array
        this->XTCFrameOffset.Add( (unsigned int)xtcFile.tellg());

        // skip some header data
        xtcFile.seekg(56, std::ios_base::cur);
        // read precision
        xtcFile.read((char*)&precision, 4);
        // change byte-order
        num = (char*)&precision;
        tmpByte = num[0]; num[0] = num[3]; num[3] = tmpByte;
        tmpByte = num[1]; num[1] = num[2]; num[2] = tmpByte;
        precision /= 10.0f;

        // get the lower bound
        xtcFile.read((char*)&minint, 12);
        // get the upper bound
        xtcFile.read((char*)&maxint, 12);
        // change byte-order
        for(i = 0; i < 3; i++ ) {
            num = (char*)&minint[i];
            tmpByte = num[0]; num[0] = num[3]; num[3] = tmpByte;
            tmpByte = num[1]; num[1] = num[2]; num[2] = tmpByte;
            num = (char*)&maxint[i];
            tmpByte = num[0]; num[0] = num[3]; num[3] = tmpByte;
            tmpByte = num[1]; num[1] = num[2]; num[2] = tmpByte;
        }

        // update the bounding box by uniting it with the last frames box
        this->bbox.Union(tmpBBox);
        // get the current frames bounding box including the atom radius
        // note: atom radius is divided by 10
        tmpBBox = vislib::math::Cuboid<float>(
            (float)minint[0] / precision - 0.3f,
            (float)minint[1] / precision - 0.3f,
            (float)minint[2] / precision - 0.3f,
            
            (float)maxint[0] / precision + 0.3f,
            (float)maxint[1] / precision + 0.3f,
            (float)maxint[2] / precision + 0.3f);

        // skip some header data
        xtcFile.seekg(4, std::ios_base::cur);

        // read size of the compressed block of data
        xtcFile.read((char*)&size, 4);
        // change byte-order
        num = (char*)&size;
        tmpByte = num[0]; num[0] = num[3]; num[3] = tmpByte;
        tmpByte = num[1]; num[1] = num[2]; num[2] = tmpByte;

        // skip the compressed block of data except for the last byte and
        // ignore the remaining bytes to prevent skipping over the end
        // of the file
        xtcFile.seekg(size-1, std::ios_base::cur);
        xtcFile.ignore(((4 - size % 4) % 4)+1);

        // add this frame to the frame count
        this->numXTCFrames++;
    }
    xtcFile.close();

    // remove the last frame
    this->XTCFrameOffset.RemoveLast();
    this->numXTCFrames--;

    vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
    "Time for parsing the XTC-file: %f",
    ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

    return true;
}

/*
 * Write all frames except for the first one from the currently loaded PDB-file
 * into a new XTC-file.
 */
void GromacsLoader::writeToXtcFile(const vislib::TString& filename) {

    std::ofstream outfile;
    unsigned int i;
    float precision = 1000.0;
    float minFloats[3];
    float maxFloats[3];

    if(data.Count() == 1) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
          "The PDB-file only contains one frame. No XTC-file has been"
          " written.");
        return;
    }

    // try to open the output-file
    outfile.open(filename, std::ios_base::binary | std::ios_base::out);

    // if the file could not be opened return
    if(!outfile) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR,
         "Could not create file.");
         return;
    }

    // get the range of value from the existing bounding box
    minFloats[0] = this->bbox.Left();   // X-coord
    maxFloats[0] = this->bbox.Right();
    minFloats[1] = this->bbox.Bottom(); // Y-coord
    maxFloats[1] = this->bbox.Top();
    minFloats[2] = this->bbox.Back();   // Z-coord
    maxFloats[2] = this->bbox.Front();

    // loop through all frames
    for(i = 0; i < data.Count(); i++) {
        data[i]->writeFrame(&outfile, precision, minFloats, maxFloats);
    }

    // close the output-file
    outfile.close();

}


