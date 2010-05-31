/*
 * IMDAtomDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "IMDAtomDataSource.h"
#include <climits>
#include "MultiParticleDataCall.h"
#include "param/BoolParam.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "utility/ColourParser.h"
#include "vislib/forceinline.h"
#include "vislib/Log.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemMessage.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Vector.h"


namespace megamol {
namespace core {
namespace moldyn {
namespace imdinternal {

    /**
     * Abstract base class for IMDAtom readers.
     *
     * Note: that no method is virtual. This allows for inline functions for
     * faster data reading.
     */
    class AbstractAtomReader {
    public:

        // No public methods are defined here (see note in class description)

    protected:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AbstractAtomReader(vislib::sys::File& file) : file(file), buf(NULL),
                validBufSize(0), pos(0) {
            this->buf = new unsigned char[BUFSIZE];
        }

        /**
         * Dtor
         */
        ~AbstractAtomReader(void) {
            // Do not close, delete, etc. the file
            ARY_SAFE_DELETE(this->buf);
        }

        /**
         * Copies a number of bytes from the input buffer to 'dst'.
         *
         * @param dst Pointer to the memory to receive the data
         * @param size The number of bytes to read
         *
         * @return 'true' on success
         */
        VISLIB_FORCEINLINE bool copyData(void *dst, unsigned int size) {
            if (this->validBufSize - this->pos >= size) {
                ::memcpy(dst, this->buf + this->pos, size);
                this->pos += size;
                return true;
            } else {
                return this->loadData(static_cast<unsigned char*>(dst), size);
            }
        }

        /**
         * Skips 'size' bytes from the input buffer
         *
         * @param size The number of bytes to skip
         *
         * @return 'true' on success
         */
        VISLIB_FORCEINLINE bool skipData(unsigned int size) {
            if (this->validBufSize - this->pos >= size) {
                this->pos += size;
                return true;
            } else {
                return this->skipDataEx(size);
            }
        }

        /**
         * Switches four bytes (endianess)
         *
         * @param ptr The pointer to the memory
         */
        VISLIB_FORCEINLINE void switch4Bytes(void *ptr) {
            char *cp = static_cast<char*>(ptr);
            char c = cp[0];
            cp[0] = cp[3];
            cp[3] = c;
            c = cp[1];
            cp[1] = cp[2];
            cp[2] = c;
        }

        /**
         * Switches eight bytes (endianess)
         *
         * @param ptr The pointer to the memory
         */
        VISLIB_FORCEINLINE void switch8Bytes(void *ptr) {
            char *cp = static_cast<char*>(ptr);
            char c = cp[0];
            cp[0] = cp[7];
            cp[7] = c;
            c = cp[1];
            cp[1] = cp[6];
            cp[6] = c;
            c = cp[2];
            cp[2] = cp[5];
            cp[5] = c;
            c = cp[3];
            cp[3] = cp[4];
            cp[4] = c;
        }

        /**
         * sifts the next ascii block
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The next ascii block
         */
        const char * sift(bool &fail) {
            static vislib::StringA ebuf;
            char * cbuf = reinterpret_cast<char *>(this->buf);

            // do we need new data?
            if (this->pos == this->validBufSize) {
                try {
                    this->validBufSize = static_cast<unsigned int>(
                        this->file.Read(this->buf, BUFSIZE));
                } catch(...) {
                    this->pos = this->validBufSize = 0;
                    fail = true;
                    return NULL;
                }
                this->pos = 0;
                if (this->validBufSize == 0) { // eof
                    fail = true;
                    return NULL;
                }
            }

            // skip white-spaces
            if (cbuf[this->pos] == 0) {
                cbuf[this->pos] = ' '; // because this 0 was proably from a
                                       // previous sift
            }
            while (vislib::CharTraitsA::IsSpace(cbuf[this->pos])) {
                this->pos++;
                if (this->pos == this->validBufSize) {
                    try {
                        this->validBufSize = static_cast<unsigned int>(
                            this->file.Read(this->buf, BUFSIZE));
                    } catch(...) {
                        this->pos = this->validBufSize = 0;
                        fail = true;
                        return NULL;
                    }
                    this->pos = 0;
                    if (this->validBufSize == 0) { // eof
                        fail = true;
                        return NULL;
                    }
                }
            }

            // collect token
            unsigned int start = this->pos; // start of the new token
            while (!vislib::CharTraitsA::IsSpace(cbuf[this->pos])) {
                this->pos++;
                if (this->pos == this->validBufSize) {
                    // running out of buffer, so we need to copy!
                    ebuf = vislib::StringA(cbuf + start, this->pos - start);

                    while (!this->file.IsEOF()) {
                        try {
                            this->validBufSize = static_cast<unsigned int>(
                                this->file.Read(this->buf, BUFSIZE));
                        } catch(...) {
                            this->pos = this->validBufSize = 0;
                            break;
                        }
                        this->pos = 0;
                        if (this->validBufSize == 0) { // eof
                            break;
                        }

                        while (!vislib::CharTraitsA::IsSpace(cbuf[this->pos])
                                && (this->pos < this->validBufSize)) {
                            this->pos++;
                        }

                        ebuf += vislib::StringA(cbuf, this->pos);
                        if (this->pos < this->validBufSize) {
                            // we did not run out of buffer -> os it's the end of the token
                            break;
                        }
                    }

                    return ebuf.PeekBuffer();
                }
            }

            // token complete in buffer (and there are remaining whitspaces)
            ASSERT(vislib::CharTraitsA::IsSpace(cbuf[this->pos]));
            this->buf[this->pos] = 0;
            return cbuf + start;
        }

    private:

        /** The size of the input buffer */
        static const unsigned int BUFSIZE = 4 * 1024;

        /**
         * Copies a number of bytes from the input buffer to 'dst'.
         *
         * @param dst Pointer to the memory to receive the data
         * @param size The number of bytes to read
         *
         * @return 'true' on success
         */
        bool loadData(unsigned char *dst, unsigned int size) {
            unsigned int rem = this->validBufSize - this->pos;
            ASSERT(size >= rem);

            ::memcpy(dst, this->buf + this->pos, rem);
            dst += rem;
            size -= rem;

            if (size > BUFSIZE) { // direct feed
                this->validBufSize = 0;
                this->pos = 0;
                try {
                    return (this->file.Read(dst, size) == size);
                } catch(...) {
                    return false;
                }
            }

            try {
                this->validBufSize = static_cast<unsigned int>(
                    this->file.Read(this->buf, BUFSIZE));

            } catch(...) {
                this->pos = this->validBufSize = 0;
                return false;
            }

            if (size > this->validBufSize) { // eof
                this->pos = this->validBufSize;
                return false;
            }

            ::memcpy(dst, this->buf, size);
            this->pos = size;
            return true;
        }

        /**
         * Skips 'size' bytes from the input buffer
         *
         * @param size The number of bytes to skip
         *
         * @return 'true' on success
         */
        bool skipDataEx(unsigned int size) {
            unsigned int rem = this->validBufSize - this->pos;
            ASSERT(size >= rem);
            size -= rem;
            this->pos = this->validBufSize;

            if (size > BUFSIZE) { // direct seek
                try {
                    this->file.Seek(size, vislib::sys::File::CURRENT);
                    return true;
                } catch(...) {
                    return false;
                }
            }

            try { // read into buffer
                this->validBufSize = static_cast<unsigned int>(
                    this->file.Read(this->buf, BUFSIZE));
            } catch(...) {
                this->pos = this->validBufSize = 0;
                return false;
            }

            if (size > this->validBufSize) { // eof
                this->pos = this->validBufSize;
                return false;
            }

            this->pos = size;
            return true;
        }

        /** The file to read from */
        vislib::sys::File& file;

        /** The input buffer */
        unsigned char *buf;

        /** The number of bytes of the input buffer that are valid */
        unsigned int validBufSize;

        /** The reading position in the input buffer */
        unsigned int pos;

    };


    /**
     * IMD Atom file reader class for the ASCII file format
     */
    class AtomReaderASCII : public AbstractAtomReader {
    public:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AtomReaderASCII(vislib::sys::File& file) : AbstractAtomReader(file) {
            // Intentionally empty
        }

        /**
         * Dtor
         */
        ~AtomReaderASCII(void) {
            // Intentionally empty
        }

        /**
         * Reads an integer from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read integer
         */
        VISLIB_FORCEINLINE UINT32 ReadInt(bool &fail) {
            const char *c = this->sift(fail);
            try {
                return static_cast<UINT32>(vislib::CharTraitsA::ParseInt(c));
            } catch(...) {
                fail = true;
            }
            return 0;
        }

        /**
         * Reads a float from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read float
         */
        VISLIB_FORCEINLINE float ReadFloat(bool &fail) {
            const char *c = this->sift(fail);
            try {
                return static_cast<float>(vislib::CharTraitsA::ParseDouble(c));
            } catch(...) {
                fail = true;
            }
            return 0.0f;
        }

        /**
         * Skips an integer in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipInt(bool &fail) {
            /*const char *c =*/ this->sift(fail);
        }

        /**
         * Skips an float in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipFloat(bool &fail) {
            /*const char *c =*/ this->sift(fail);
        }

    };


    /**
     * IMD Atom file reader class for the float file format
     */
    class AtomReaderFloat : public AbstractAtomReader {
    public:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AtomReaderFloat(vislib::sys::File& file) : AbstractAtomReader(file) {
            // Intentionally empty
        }

        /**
         * Dtor
         */
        ~AtomReaderFloat(void) {
            // Intentionally empty
        }

        /**
         * Reads an integer from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read integer
         */
        VISLIB_FORCEINLINE UINT32 ReadInt(bool &fail) {
            UINT32 i;
            if (!this->copyData(&i, 4)) {
                fail = true;
            }
            return i;
        }

        /**
         * Reads a float from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read float
         */
        VISLIB_FORCEINLINE float ReadFloat(bool &fail) {
            float f;
            if (!this->copyData(&f, 4)) {
                fail = true;
            }
            return f;
        }

        /**
         * Skips an integer in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipInt(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

        /**
         * Skips an float in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipFloat(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

    };


    /**
     * IMD Atom file reader class for the double file format
     */
    class AtomReaderDouble : public AbstractAtomReader {
    public:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AtomReaderDouble(vislib::sys::File& file) : AbstractAtomReader(file) {
            // Intentionally empty
        }

        /**
         * Dtor
         */
        ~AtomReaderDouble(void) {
            // Intentionally empty
        }

        /**
         * Reads an integer from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read integer
         */
        VISLIB_FORCEINLINE UINT32 ReadInt(bool &fail) {
            UINT32 i;
            if (!this->copyData(&i, 4)) {
                fail = true;
            }
            return i;
        }

        /**
         * Reads a float from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read float
         */
        VISLIB_FORCEINLINE float ReadFloat(bool &fail) {
            double d;
            if (!this->copyData(&d, 8)) {
                fail = true;
            }
            return static_cast<float>(d);
        }

        /**
         * Skips an integer in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipInt(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

        /**
         * Skips an float in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipFloat(bool &fail) {
            if (!this->skipData(8)) fail = true;
        }

    };


    /**
     * IMD Atom file reader class for the float file format with byte switch
     */
    class AtomReaderFloatSwitched : public AbstractAtomReader {
    public:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AtomReaderFloatSwitched(vislib::sys::File& file) : AbstractAtomReader(file) {
            // Intentionally empty
        }

        /**
         * Dtor
         */
        ~AtomReaderFloatSwitched(void) {
            // Intentionally empty
        }

        /**
         * Reads an integer from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read integer
         */
        VISLIB_FORCEINLINE UINT32 ReadInt(bool &fail) {
            UINT32 i;
            if (!this->copyData(&i, 4)) {
                fail = true;
            } else {
                this->switch4Bytes(&i);
            }
            return i;
        }

        /**
         * Reads a float from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read float
         */
        VISLIB_FORCEINLINE float ReadFloat(bool &fail) {
            float f;
            if (!this->copyData(&f, 4)) {
                fail = true;
            } else {
                this->switch4Bytes(&f);
            }
            return f;
        }

        /**
         * Skips an integer in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipInt(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

        /**
         * Skips an float in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipFloat(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

    };


    /**
     * IMD Atom file reader class for the double file format with byte switch
     */
    class AtomReaderDoubleSwitched : public AbstractAtomReader {
    public:

        /**
         * Ctor
         *
         * @param file The file to read from
         */
        AtomReaderDoubleSwitched(vislib::sys::File& file) : AbstractAtomReader(file) {
            // Intentionally empty
        }

        /**
         * Dtor
         */
        ~AtomReaderDoubleSwitched(void) {
            // Intentionally empty
        }

        /**
         * Reads an integer from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read integer
         */
        VISLIB_FORCEINLINE UINT32 ReadInt(bool &fail) {
            UINT32 i;
            if (!this->copyData(&i, 4)) {
                fail = true;
            } else {
                this->switch4Bytes(&i);
            }
            return i;
        }

        /**
         * Reads a float from the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         *
         * @return The read float
         */
        VISLIB_FORCEINLINE float ReadFloat(bool &fail) {
            double d;
            if (!this->copyData(&d, 8)) {
                fail = true;
            } else {
                this->switch8Bytes(&d);
            }
            return static_cast<float>(d);
        }

        /**
         * Skips an integer in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipInt(bool &fail) {
            if (!this->skipData(4)) fail = true;
        }

        /**
         * Skips an float in the input data
         *
         * @param fail The fail flag is not changed if the method succeeds.
         *             If the method fails the flag is set to 'true'.
         */
        VISLIB_FORCEINLINE void SkipFloat(bool &fail) {
            if (!this->skipData(8)) fail = true;
        }

    };

} /* end namespace imdinternal */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

using namespace megamol::core;


/*
 * moldyn::IMDAtomDataSource::IMDAtomDataSource
 */
moldyn::IMDAtomDataSource::IMDAtomDataSource(void) : Module(),
        filenameSlot("filename", "The path of the IMD file to read"),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        radiusSlot("radius", "The radius to be used for the data"),
        colourModeSlot("colmode", "The colouring option"),
        colourSlot("col", "The default colour to be used for the \"const\" colour mode"),
        colourColumnSlot("colcolumn", "The data column used for the \"column\" colour mode"),
        autoColumnRangeSlot("autoColumnRange", "Whether or not to automatically calculate the column value range"),
        minColumnValSlot("minColumnValue", "The minimum value for the colour mapping of the column"),
        maxColumnValSlot("maxColumnValue", "The maximum value for the colour mapping of the column"),
        posData(), colData(), headerMinX(0.0f), headerMinY(0.0f),
        headerMinZ(0.0f), headerMaxX(1.0f), headerMaxY(1.0f),
        headerMaxZ(1.0f), minX(0.0f), minY(0.0f), minZ(0.0f), maxX(1.0f),
        maxY(1.0f), maxZ(1.0f), minC(0.0f), maxC(1.0f), datahash(0) {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetData",
        &IMDAtomDataSource::getDataCallback);
    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetExtent",
        &IMDAtomDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->radiusSlot << new param::FloatParam(0.5f, 0.0000001f);
    this->MakeSlotAvailable(&this->radiusSlot);

    param::EnumParam * cm = new param::EnumParam(1);
    cm->SetTypePair(0, "const");
    cm->SetTypePair(1, "column");
    this->colourModeSlot << cm;
    this->MakeSlotAvailable(&this->colourModeSlot);

    this->colourSlot << new param::StringParam("Silver");
    this->colourSlot.ForceSetDirty();
    this->MakeSlotAvailable(&this->colourSlot);

    this->colourColumnSlot << new param::StringParam("0");
    this->MakeSlotAvailable(&this->colourColumnSlot);

    this->autoColumnRangeSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->autoColumnRangeSlot);

    this->minColumnValSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->minColumnValSlot);

    this->maxColumnValSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->maxColumnValSlot);

}


/*
 * moldyn::IMDAtomDataSource::~IMDAtomDataSource
 */
moldyn::IMDAtomDataSource::~IMDAtomDataSource(void) {
    this->Release();
}


/*
 * moldyn::IMDAtomDataSource::create
 */
bool moldyn::IMDAtomDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::IMDAtomDataSource::release
 */
void moldyn::IMDAtomDataSource::release(void) {
    this->clear();
}


/*
 * moldyn::IMDAtomDataSource::getDataCallback
 */
bool moldyn::IMDAtomDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;
    this->assertData();

    mpdc->SetFrameID(0);
    mpdc->SetDataHash(this->datahash);
    mpdc->SetParticleListCount(1); // For the moment
    if (this->colourSlot.IsDirty()) {
        this->colourSlot.ResetDirty();
        float r, g, b;
        if (utility::ColourParser::FromString(this->colourSlot.Param<param::StringParam>()->Value(), r, g, b)) {
            this->defCol[0] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(r * 255.0f), 0, 255));
            this->defCol[1] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(g * 255.0f), 0, 255));
            this->defCol[2] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(b * 255.0f), 0, 255));
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to parse default colour \"%s\"\n",
                vislib::StringA(this->colourSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        }
    }
    mpdc->AccessParticles(0).SetGlobalColour(
        this->defCol[0], this->defCol[1], this->defCol[2]);
    mpdc->AccessParticles(0).SetGlobalRadius(
        this->radiusSlot.Param<param::FloatParam>()->Value());

    mpdc->AccessParticles(0).SetCount(
        this->posData.GetSize() / (3 * sizeof(float)));

    int colMode = this->colourModeSlot.Param<param::EnumParam>()->Value();
    if ((colMode == 1) && (this->colData.GetSize() == 0)) {
        colMode = 0;
    }
    switch (colMode) {
        case 0:
            mpdc->AccessParticles(0).SetColourData(
                MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            break;
        case 1:
            if (this->autoColumnRangeSlot.Param<param::BoolParam>()->Value()) {
                mpdc->AccessParticles(0).SetColourMapIndexValues(this->minC, this->maxC);
            } else {
                mpdc->AccessParticles(0).SetColourMapIndexValues(
                    this->minColumnValSlot.Param<param::FloatParam>()->Value(),
                    this->maxColumnValSlot.Param<param::FloatParam>()->Value());
            }
            mpdc->AccessParticles(0).SetColourData(
                MultiParticleDataCall::Particles::COLDATA_FLOAT_I, this->colData.As<void>());
            break;
        default:
            mpdc->AccessParticles(0).SetColourData( // some internal error
                MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            break;
    }

    if (!this->posData.IsEmpty()) {
        mpdc->AccessParticles(0).SetVertexData(
            MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
            this->posData.As<void>());
    } else {
        mpdc->AccessParticles(0).SetVertexData(
            MultiParticleDataCall::Particles::VERTDATA_NONE, NULL);
    }
    mpdc->SetUnlocker(NULL);

    return true;
}


/*
 * moldyn::IMDAtomDataSource::getExtentCallback
 */
bool moldyn::IMDAtomDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;
    this->assertData();

    mpdc->SetDataHash(this->datahash);
    float rad = this->radiusSlot.Param<param::FloatParam>()->Value();

    mpdc->SetFrameCount(1);
    mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(
        this->headerMinX - rad, this->headerMinY - rad, this->headerMinZ - rad,
        this->headerMaxX + rad, this->headerMaxY + rad, this->headerMaxZ + rad);
    mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(
        this->minX - rad, this->minY - rad, this->minZ - rad,
        this->maxX + rad, this->maxY + rad, this->maxZ + rad);

    return true;
}


/*
 * moldyn::IMDAtomDataSource::clear
 */
void moldyn::IMDAtomDataSource::clear(void) {
    this->posData.EnforceSize(0);
    this->colData.EnforceSize(0);
    this->headerMinX = this->headerMinY = this->headerMinZ = 0.0f;
    this->headerMaxX = this->headerMaxY = this->headerMaxZ = 1.0f;
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash++;
}


/*
 * moldyn::IMDAtomDataSource::assertData
 */
void moldyn::IMDAtomDataSource::assertData(void) {
    using vislib::sys::Log;
    if (!this->filenameSlot.IsDirty() && !this->colourModeSlot.IsDirty()
            && !this->colourColumnSlot.IsDirty()) return;
    this->filenameSlot.ResetDirty();
    this->colourModeSlot.ResetDirty();
    this->colourColumnSlot.ResetDirty();

    this->clear();

    vislib::sys::MemmappedFile file;
    vislib::TString filename = this->filenameSlot.Param<param::FilePathParam>()->Value();
    //this->datahash = static_cast<SIZE_T>(filename.HashCode());
    if (!file.Open(filename, vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to open imd file %s\n", vislib::StringA(
            this->filenameSlot.Param<param::FilePathParam>()->Value()).PeekBuffer());
        return;
    }

    HeaderData header;
    if (!this->readHeader(file, header)) {
        // error already logged
        file.Close();
        return;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
        "IMDAtom with %d data colums:\n", static_cast<int>(header.captions.Count()));
    for (SIZE_T i = 0; i < header.captions.Count(); i++) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
            "\t%s\n", header.captions[i].PeekBuffer());
    }

    UINT32 endianTestInt = 0x12345678;
    UINT8 endianTestBytes[4];
    ::memcpy(endianTestBytes, &endianTestInt, 4);
    bool machineBigEndian = ((endianTestBytes[0] == 0x78)
        && (endianTestBytes[1] == 0x56)
        && (endianTestBytes[2] == 0x34)
        && (endianTestBytes[3] == 0x12));

    vislib::RawStorageWriter posWriter(this->posData, 0, 0, 10 * 1024 * 1024);
    vislib::RawStorageWriter colWriter(this->colData, 0, 0, 10 * 1024 * 1024);

    bool retval = false;
    switch (header.format) {
        case 'A': // ASCII
            retval = this->readData<imdinternal::AtomReaderASCII>(file, header, posWriter, colWriter);
            break;
        case 'B': // binary, big endian, double
            retval = (machineBigEndian)
                ? this->readData<imdinternal::AtomReaderDoubleSwitched>(file, header, posWriter, colWriter)
                : this->readData<imdinternal::AtomReaderDouble>(file, header, posWriter, colWriter);
            break;
        case 'b': // binary, big endian, float
            retval = (machineBigEndian)
                ? this->readData<imdinternal::AtomReaderFloatSwitched>(file, header, posWriter, colWriter)
                : this->readData<imdinternal::AtomReaderFloat>(file, header, posWriter, colWriter);
            break;
        case 'L': // binary, little endian, double
            retval = (machineBigEndian)
                ? this->readData<imdinternal::AtomReaderDouble>(file, header, posWriter, colWriter)
                : this->readData<imdinternal::AtomReaderDoubleSwitched>(file, header, posWriter, colWriter);
            break;
        case 'l': // binary, little endian float
            retval = (machineBigEndian)
                ? this->readData<imdinternal::AtomReaderFloat>(file, header, posWriter, colWriter)
                : this->readData<imdinternal::AtomReaderFloatSwitched>(file, header, posWriter, colWriter);
            break;
        default:
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to read imd file: Illegal format\n");
            break;
    }

    if (retval) {
        this->posData.EnforceSize(posWriter.End(), true);
        this->colData.EnforceSize(colWriter.End(), true);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%d Atoms loaded\n", this->posData.GetSize() / (sizeof(float) * 3));
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
            "Data bounding box = (%f, %f, %f) ... (%f, %f, %f)\n", 
            this->minX, this->minY, this->minZ,
            this->maxX, this->maxY, this->maxZ);

        //this->datahash = (this->datahash << (sizeof(SIZE_T) / 2))
        //    || (this->datahash >> (sizeof(SIZE_T) / 2));
        //this->datahash ^= this->posData.GetSize();
        this->datahash++;

        // All parameters must influence the data hash

    } else {
        // error already logged
        this->posData.EnforceSize(0, true);
        this->colData.EnforceSize(0, true);
        this->datahash = 0;
    }

    file.Close();
}


/*
 * moldyn::IMDAtomDataSource::readHeader
 */
bool moldyn::IMDAtomDataSource::readHeader(vislib::sys::File& file,
        moldyn::IMDAtomDataSource::HeaderData& header) {
    using vislib::sys::Log;
    using vislib::sys::File;
    vislib::StringA line;
    bool windowsNewline = false;
    char nlb[2];

    try {
        file.SeekToBegin();
        line = vislib::sys::ReadLineFromFileA(file);
        if (line[0] != '#') {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Failed to parse IMD header: Illegal first line character %c\n", line[0]);
            return false;
        }

        file.Seek(-2, File::CURRENT);
        file.Read(nlb, 2);
        file.SeekToBegin();
        if ((nlb[0] == '\r') && (nlb[1] == '\n')) {
            windowsNewline = true; // used for the '#E' line
        }

        header.captions.Clear();

        int warnCnt = 0;
        int linePos = 0;
        while (!file.IsEOF()) {
            line = vislib::sys::ReadLineFromFileA(file);
            linePos++;

            if (line[0] != '#') {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Line %d has illegal first character: %c\n", linePos, line[0]);
                warnCnt++;
                if (warnCnt == 10) {
                    throw new vislib::Exception("Too many warnings", __FILE__, __LINE__);
                } else {
                    continue;
                }
            }

            switch (line[1]) {
                case '#': break; // comment line
                case 'F': { // format line
                    line.Replace('\t', ' ');
                    vislib::Array<vislib::StringA> itemz
                        = vislib::StringTokeniserA::Split(line, ' ', true);
                    if (itemz.Count() != 8) {
                        vislib::StringA msg("Illegal format line (not 7 fields): ");
                        msg.Append(line);
                        throw new vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                    }

                    header.format = itemz[1][0];
                    if (((header.format != 'A') && (header.format != 'B') && (header.format != 'b') && 
                           (header.format != 'L') && (header.format != 'l')) || (itemz[1].Length() != 1)) {
                        vislib::StringA msg("Illegal format: ");
                        msg.Append(itemz[1]);
                        throw new vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                    }

                    header.id = (vislib::CharTraitsA::ParseInt(itemz[2]) != 0);
                    header.type = (vislib::CharTraitsA::ParseInt(itemz[3]) != 0);
                    header.mass = (vislib::CharTraitsA::ParseInt(itemz[4]) != 0);
                    header.pos = vislib::CharTraitsA::ParseInt(itemz[5]);
                    header.vel = vislib::CharTraitsA::ParseInt(itemz[6]);
                    header.dat = vislib::CharTraitsA::ParseInt(itemz[7]);

                    if ((header.pos != 2) && (header.pos != 3)) {
                        vislib::StringA msg("Illegal position vector size: ");
                        msg.Append(itemz[5]);
                        throw new vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                    }

                    if ((header.vel > 0) && (header.vel != header.pos)) {
                        vislib::StringA msg("Illegal velocity vector size: ");
                        msg.Append(itemz[6]);
                        throw new vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                    }

                } break;
                case 'C': // caption line
                    line.Replace('\t', ' ');
                    header.captions = vislib::StringTokeniserA::Split(line, ' ', true);
                    header.captions.Erase(0, 1);
                    break;
                case 'X': { // bounding box x line
                    line.Replace('\t', ' ');
                    vislib::Array<vislib::StringA> itemz
                        = vislib::StringTokeniserA::Split(line, ' ', true);
                    if (itemz.Count() < 3) {
                        throw new vislib::Exception("Illegal bounding box x vector",
                            __FILE__, __LINE__);
                    }
                    vislib::math::Vector<double, 3> vec(
                        vislib::CharTraitsA::ParseDouble(itemz[1]),
                        vislib::CharTraitsA::ParseDouble(itemz[2]), 0.0);
                    if (itemz.Count() >= 4) {
                        vec.SetZ(vislib::CharTraitsA::ParseDouble(itemz[3]));
                    }
                    this->headerMaxX = static_cast<float>(vec.Length());
                } break;
                case 'Y': { // bounding box y line
                    line.Replace('\t', ' ');
                    vislib::Array<vislib::StringA> itemz
                        = vislib::StringTokeniserA::Split(line, ' ', true);
                    if (itemz.Count() < 3) {
                        throw new vislib::Exception("Illegal bounding box y vector",
                            __FILE__, __LINE__);
                    }
                    vislib::math::Vector<double, 3> vec(
                        vislib::CharTraitsA::ParseDouble(itemz[1]),
                        vislib::CharTraitsA::ParseDouble(itemz[2]), 0.0);
                    if (itemz.Count() >= 4) {
                        vec.SetZ(vislib::CharTraitsA::ParseDouble(itemz[3]));
                    }
                    this->headerMaxY = static_cast<float>(vec.Length());
                } break;
                case 'Z': { // bounding box z line
                    line.Replace('\t', ' ');
                    vislib::Array<vislib::StringA> itemz
                        = vislib::StringTokeniserA::Split(line, ' ', true);
                    if (itemz.Count() < 4) {
                        throw new vislib::Exception("Illegal bounding box z vector",
                            __FILE__, __LINE__);
                    }
                    vislib::math::Vector<double, 3> vec(
                        vislib::CharTraitsA::ParseDouble(itemz[1]),
                        vislib::CharTraitsA::ParseDouble(itemz[2]),
                        vislib::CharTraitsA::ParseDouble(itemz[3]));
                    this->headerMaxZ = static_cast<float>(vec.Length());
                } break;
                case 'E': // end header line

                    if (header.format != 'A') {
                        // fix a newline at the end (should never happen)
                        file.Seek(-2, File::CURRENT);
                        file.Read(nlb, 2);
                        if ((nlb[0] == '\r') && (nlb[1] == '\n') && !windowsNewline) {
                            file.Seek(-1, File::CURRENT);
                        }
                    }

                    if (!header.captions.IsEmpty()) {
                        int cnt = 0;
                        if (header.id) cnt++;
                        if (header.type) cnt++;
                        if (header.mass) cnt++;
                        cnt += header.pos + header.vel + header.dat;
                        int hcnt = static_cast<int>(header.captions.Count());
                        if (hcnt < cnt) {
                            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                                "Too few data column captions specified (%d instead of %d)",
                                hcnt, cnt);
                            for (; hcnt < cnt; hcnt++) {
                                header.captions.Add("unnamed");
                            }
                        } else if (hcnt > cnt) {
                            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                                "Too many data column captions specified (%d instead of %d)",
                                hcnt, cnt);
                            header.captions.Erase(cnt, hcnt - cnt);
                        }
                    }

                    return true;
            }

        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to parse IMD header: unexpected end of file\n");

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to parse IMD header: %s\n", ex.GetMsgA());
    } catch(...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to parse IMD header: unexpected exception\n");
    }

    return false;
}


/*
 * moldyn::IMDAtomDataSource::readData
 */
template<typename T>
bool moldyn::IMDAtomDataSource::readData(vislib::sys::File& file,
        const moldyn::IMDAtomDataSource::HeaderData& header,
        vislib::RawStorageWriter& pos, vislib::RawStorageWriter& col) {
    T reader(file);
    bool fail = false;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    bool first = true;
    float c = 0.0f;
    unsigned int column;
    unsigned int colcolumn = UINT_MAX;

    if (this->colourModeSlot.Param<param::EnumParam>()->Value() == 1) {
        // column colouring mode
        vislib::StringA colcolname(this->colourColumnSlot.Param<param::StringParam>()->Value());

        // 1. exact match
        for (SIZE_T i = 0; i < header.captions.Count(); i++) {
            if (header.captions[i].Equals(colcolname)) {
                colcolumn = static_cast<unsigned int>(i);
                break;
            }
        }

        if (colcolumn == UINT_MAX) {
            // 2. caseless match
            for (SIZE_T i = 0; i < header.captions.Count(); i++) {
                if (header.captions[i].Equals(colcolname, false)) {
                    colcolumn = static_cast<unsigned int>(i);
                    break;
                }
            }
        }

        if (colcolumn == UINT_MAX) {
            // 3. index
            try {
                colcolumn = vislib::CharTraitsA::ParseInt(colcolname);
                if (colcolumn >= static_cast<unsigned int>(header.captions.Count())) {
                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                        "The parsed colouring column index is out of range (%u not in 0..%d)\n",
                        colcolumn, static_cast<int>(header.captions.Count()) - 1);
                    colcolumn = UINT_MAX;
                }
            } catch(...) {
                colcolumn = UINT_MAX;
            }
        }

        if (colcolumn == UINT_MAX) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Failed to parse colour column selection: %s\n",
                colcolname.PeekBuffer());
        }
    }

    while (!fail) {
        column = 0;

        if (header.id) {
            if (column == colcolumn) {
                c = static_cast<float>(reader.ReadInt(fail));
            } else {
                reader.SkipInt(fail);
            }
            column++;
        }
        if (header.type) {
            if (column == colcolumn) {
                c = static_cast<float>(reader.ReadInt(fail));
            } else {
                reader.SkipInt(fail);
            }
            column++;
        }
        if (header.mass) {
            if (column == colcolumn) {
                c = reader.ReadFloat(fail);
            } else {
                reader.SkipFloat(fail);
            }
            column++;
        }
        for (int i = 0; i < header.pos; i++) {
            if (i == 0) {
                x = reader.ReadFloat(fail);
                if (column == colcolumn) c = x;
            }
            if (i == 1) {
                y = reader.ReadFloat(fail);
                if (column == colcolumn) c = y;
            }
            if (i == 2) {
                z = reader.ReadFloat(fail);
                if (column == colcolumn) c = z;
            }
            if (i >= 3) {
                if (column == colcolumn) {
                    c = reader.ReadFloat(fail);
                } else {
                    reader.SkipFloat(fail);
                }
            }
            column++;
        }
        for (int i = 0; i < header.vel; i++) {
            if (column == colcolumn) {
                c = reader.ReadFloat(fail);
            } else {
                reader.SkipFloat(fail);
            }
            column++;
        }
        for (int i = 0; i < header.dat; i++) {
            if (column == colcolumn) {
                c = reader.ReadFloat(fail);
            } else {
                reader.SkipFloat(fail);
            }
            column++;
        }

        if (!fail) {
            if (!first) {
                if (this->minX > x) this->minX = x; else
                if (this->maxX < x) this->maxX = x;
                if (this->minY > y) this->minY = y; else
                if (this->maxY < y) this->maxY = y;
                if (this->minZ > z) this->minZ = z; else
                if (this->maxZ < z) this->maxZ = z;
            } else {
                first = false;
                this->minX = this->maxX = x;
                this->minY = this->maxY = y;
                this->minZ = this->maxZ = z;
                this->minC = this->maxC = c;
            }
            pos << x << y << z;
            if (colcolumn != UINT_MAX) {
                col << c;
                if (this->minC > c) this->minC = c; else
                if (this->maxC < c) this->maxC = c;
            }
        }
    }

    return !first;
}
