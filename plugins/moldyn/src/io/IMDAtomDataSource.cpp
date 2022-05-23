/*
 * IMDAtomDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "io/IMDAtomDataSource.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include "vislib/Array.h"
#include "vislib/PtrArray.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/forceinline.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/SystemMessage.h"
#include "vislib/sys/sysfunctions.h"
#include <climits>


namespace {

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
    AbstractAtomReader(vislib::sys::File& file) : file(file), buf(NULL), validBufSize(0), pos(0) {
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
    VISLIB_FORCEINLINE bool copyData(void* dst, unsigned int size) {
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
    VISLIB_FORCEINLINE void switch4Bytes(void* ptr) {
        char* cp = static_cast<char*>(ptr);
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
    VISLIB_FORCEINLINE void switch8Bytes(void* ptr) {
        char* cp = static_cast<char*>(ptr);
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
    const char* sift(bool& fail) {
        static vislib::StringA ebuf;
        char* cbuf = reinterpret_cast<char*>(this->buf);

        // do we need new data?
        if (this->pos == this->validBufSize) {
            try {
                this->validBufSize = static_cast<unsigned int>(this->file.Read(this->buf, BUFSIZE));
            } catch (...) {
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
                    this->validBufSize = static_cast<unsigned int>(this->file.Read(this->buf, BUFSIZE));
                } catch (...) {
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
                        this->validBufSize = static_cast<unsigned int>(this->file.Read(this->buf, BUFSIZE));
                    } catch (...) {
                        this->pos = this->validBufSize = 0;
                        break;
                    }
                    this->pos = 0;
                    if (this->validBufSize == 0) { // eof
                        break;
                    }

                    while (!vislib::CharTraitsA::IsSpace(cbuf[this->pos]) && (this->pos < this->validBufSize)) {
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
    bool loadData(unsigned char* dst, unsigned int size) {
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
            } catch (...) { return false; }
        }

        try {
            this->validBufSize = static_cast<unsigned int>(this->file.Read(this->buf, BUFSIZE));

        } catch (...) {
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
            } catch (...) { return false; }
        }

        try { // read into buffer
            this->validBufSize = static_cast<unsigned int>(this->file.Read(this->buf, BUFSIZE));
        } catch (...) {
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
    unsigned char* buf;

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
    VISLIB_FORCEINLINE UINT32 ReadInt(bool& fail) {
        const char* c = this->sift(fail);
        try {
            return static_cast<UINT32>(vislib::CharTraitsA::ParseInt(c));
        } catch (...) { fail = true; }
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
    VISLIB_FORCEINLINE float ReadFloat(bool& fail) {
        const char* c = this->sift(fail);
        try {
            return static_cast<float>(vislib::CharTraitsA::ParseDouble(c));
        } catch (...) { fail = true; }
        return 0.0f;
    }

    /**
     * Skips an integer in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipInt(bool& fail) { /*const char *c =*/
        this->sift(fail);
    }

    /**
     * Skips an float in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipFloat(bool& fail) { /*const char *c =*/
        this->sift(fail);
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
    VISLIB_FORCEINLINE UINT32 ReadInt(bool& fail) {
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
    VISLIB_FORCEINLINE float ReadFloat(bool& fail) {
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
    VISLIB_FORCEINLINE void SkipInt(bool& fail) {
        if (!this->skipData(4))
            fail = true;
    }

    /**
     * Skips an float in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipFloat(bool& fail) {
        if (!this->skipData(4))
            fail = true;
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
    VISLIB_FORCEINLINE UINT32 ReadInt(bool& fail) {
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
    VISLIB_FORCEINLINE float ReadFloat(bool& fail) {
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
    VISLIB_FORCEINLINE void SkipInt(bool& fail) {
        if (!this->skipData(4))
            fail = true;
    }

    /**
     * Skips an float in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipFloat(bool& fail) {
        if (!this->skipData(8))
            fail = true;
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
    VISLIB_FORCEINLINE UINT32 ReadInt(bool& fail) {
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
    VISLIB_FORCEINLINE float ReadFloat(bool& fail) {
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
    VISLIB_FORCEINLINE void SkipInt(bool& fail) {
        if (!this->skipData(4))
            fail = true;
    }

    /**
     * Skips an float in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipFloat(bool& fail) {
        if (!this->skipData(4))
            fail = true;
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
    VISLIB_FORCEINLINE UINT32 ReadInt(bool& fail) {
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
    VISLIB_FORCEINLINE float ReadFloat(bool& fail) {
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
    VISLIB_FORCEINLINE void SkipInt(bool& fail) {
        if (!this->skipData(4))
            fail = true;
    }

    /**
     * Skips an float in the input data
     *
     * @param fail The fail flag is not changed if the method succeeds.
     *             If the method fails the flag is set to 'true'.
     */
    VISLIB_FORCEINLINE void SkipFloat(bool& fail) {
        if (!this->skipData(8))
            fail = true;
    }
};

} /* end anonymous namespace */

using namespace megamol;
using namespace megamol::moldyn::io;


/*
 * IMDAtomDataSource::FileFormatAutoDetect
 */
float IMDAtomDataSource::FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize) {
    vislib::StringA line(reinterpret_cast<const char*>(data), static_cast<unsigned int>(dataSize));
    line.Replace('\t', ' ');
    vislib::Array<vislib::StringA> lines = vislib::StringTokeniserA::Split(line, "\n", true);
    for (SIZE_T i = 0; i < lines.Count(); i++) {
        if (!lines[i].StartsWith("#F "))
            continue;
        vislib::Array<vislib::StringA> tokens = vislib::StringTokeniserA::Split(lines[i], ' ', true);
        if (tokens.Count() != 8)
            continue;

        // now I'm almost sure
        if (tokens[1].Equals("A") || tokens[1].Equals("B") || tokens[1].Equals("b") || tokens[1].Equals("L") ||
            tokens[1].Equals("l")) {
            // now I'm sure
            return 1.0f;
        }
    }

    // I'm not sure at all
    return 0.0f;
}


/*
 * IMDAtomDataSource::IMDAtomDataSource
 */
IMDAtomDataSource::IMDAtomDataSource(void)
        : core::Module()
        , filenameSlot("filename", "The path of the IMD file to read")
        , bboxEnabledSlot("bbox::enable", "")
        , bboxMinSlot("bbox::min", "")
        , bboxMaxSlot("bbox::max", "")
        , getDataSlot("getdata", "The slot exposing the loaded data")
        , radiusSlot("radius", "The radius to be used for the data")
        , colourModeSlot("colmode", "The colouring option")
        , colourSlot("col", "The default colour to be used for the \"const\" colour mode")
        , colourColumnSlot("colcolumn", "The data column used for the \"column\" colour mode")
        , autoColumnRangeSlot("autoColumnRange", "Whether or not to automatically calculate the column value range")
        , minColumnValSlot("minColumnValue", "The minimum value for the colour mapping of the column")
        , maxColumnValSlot("maxColumnValue", "The maximum value for the colour mapping of the column")
        , typeColumnSlot("typecolumn", "The data column used for determining the particle type")
        , posXFilterNow("filter::posXFilter", "")
        , posXFilter("filter::posX", "")
        , posXMinFilter("filter::posXMin", "")
        , posXMaxFilter("filter::posXMax", "")
        , splitLoadDiredDataSlot("dir::split", "")
        , dirXColNameSlot("dir::x", "")
        , dirYColNameSlot("dir::y", "")
        , dirZColNameSlot("dir::z", "")
        , dircolourModeSlot("dir::colmode", "The colouring option")
        , dircolourSlot("dir::col", "The default colour to be used for the \"const\" colour mode")
        , dircolourColumnSlot("dir::colcolumn", "The data column used for the \"column\" colour mode")
        , dirautoColumnRangeSlot(
              "dir::autoColumnRange", "Whether or not to automatically calculate the column value range")
        , dirminColumnValSlot("dir::minColumnValue", "The minimum value for the colour mapping of the column")
        , dirmaxColumnValSlot("dir::maxColumnValue", "The maximum value for the colour mapping of the column")
        , dirradiusSlot("dir::radius", "The radius to be used for the data")
        , dirNormDirSlot("dir::normalise", "")
        , posData()
        , colData()
        , headerMinX(0.0f)
        , headerMinY(0.0f)
        , headerMinZ(0.0f)
        , headerMaxX(1.0f)
        , headerMaxY(1.0f)
        , headerMaxZ(1.0f)
        , minX(0.0f)
        , minY(0.0f)
        , minZ(0.0f)
        , maxX(1.0f)
        , maxY(1.0f)
        , maxZ(1.0f)
        , minC()
        , maxC()
        , datahash(0)
        , allDirData()
        , typeData() {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->bboxEnabledSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->bboxEnabledSlot);
    vislib::math::Vector<float, 3> vMin;
    vMin.Set(0.0f, 0.0f, 0.0f);
    vislib::math::Vector<float, 3> vMax;
    vMin.Set(1.0f, 1.0f, 1.0f);
    this->bboxMinSlot << new core::param::Vector3fParam(vMin);
    this->MakeSlotAvailable(&this->bboxMinSlot);
    this->bboxMaxSlot << new core::param::Vector3fParam(vMax);
    this->MakeSlotAvailable(&this->bboxMaxSlot);

    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetData", &IMDAtomDataSource::getDataCallback);
    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &IMDAtomDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->radiusSlot << new core::param::FloatParam(0.5f, 0.0000001f);
    this->MakeSlotAvailable(&this->radiusSlot);

    core::param::EnumParam* cm = new core::param::EnumParam(1);
    cm->SetTypePair(0, "const");
    cm->SetTypePair(1, "column");
    this->colourModeSlot << cm;
    this->MakeSlotAvailable(&this->colourModeSlot);

    this->colourSlot << new core::param::StringParam("Silver");
    this->colourSlot.ForceSetDirty();
    this->MakeSlotAvailable(&this->colourSlot);

    this->colourColumnSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->colourColumnSlot);

    this->typeColumnSlot << new core::param::StringParam("1");
    this->MakeSlotAvailable(&this->typeColumnSlot);

    this->autoColumnRangeSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->autoColumnRangeSlot);

    this->minColumnValSlot << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->minColumnValSlot);

    this->maxColumnValSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->maxColumnValSlot);

    this->posXFilterNow << new core::param::ButtonParam(core::view::Key::KEY_F);
    this->posXFilterNow.SetUpdateCallback(&IMDAtomDataSource::posXFilterUpdate);
    this->MakeSlotAvailable(&this->posXFilterNow);
    this->posXFilter << new core::param::BoolParam(false);
    this->posXFilter.SetUpdateCallback(&IMDAtomDataSource::posXFilterUpdate);
    this->MakeSlotAvailable(&this->posXFilter);
    this->posXMinFilter << new core::param::FloatParam(0.0f);
    this->posXMinFilter.SetUpdateCallback(&IMDAtomDataSource::posXFilterUpdate);
    this->MakeSlotAvailable(&this->posXMinFilter);
    this->posXMaxFilter << new core::param::FloatParam(1.0f);
    this->posXMaxFilter.SetUpdateCallback(&IMDAtomDataSource::posXFilterUpdate);
    this->MakeSlotAvailable(&this->posXMaxFilter);

    this->splitLoadDiredDataSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->splitLoadDiredDataSlot);
    this->dirXColNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->dirXColNameSlot);
    this->dirYColNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->dirYColNameSlot);
    this->dirZColNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->dirZColNameSlot);
    cm = new core::param::EnumParam(1);
    cm->SetTypePair(0, "const");
    cm->SetTypePair(1, "column");
    cm->SetTypePair(2, "direction");
    this->dircolourModeSlot << cm;
    this->MakeSlotAvailable(&this->dircolourModeSlot);
    this->dircolourSlot << new core::param::StringParam("Silver");
    this->dircolourSlot.ForceSetDirty();
    this->MakeSlotAvailable(&this->dircolourSlot);
    this->dircolourColumnSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->dircolourColumnSlot);
    this->dirautoColumnRangeSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->dirautoColumnRangeSlot);
    this->dirminColumnValSlot << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->dirminColumnValSlot);
    this->dirmaxColumnValSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->dirmaxColumnValSlot);
    this->dirradiusSlot << new core::param::FloatParam(0.5f, 0.0000001f);
    this->MakeSlotAvailable(&this->dirradiusSlot);
    this->dirNormDirSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->dirNormDirSlot);
}


/*
 * IMDAtomDataSource::~IMDAtomDataSource
 */
IMDAtomDataSource::~IMDAtomDataSource(void) {
    this->Release();
}


/*
 * IMDAtomDataSource::create
 */
bool IMDAtomDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * IMDAtomDataSource::release
 */
void IMDAtomDataSource::release(void) {
    this->clear();
}


/*
 * IMDAtomDataSource::getDataCallback
 */
bool IMDAtomDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL)
        return false;
    this->assertData();

    if (this->colourSlot.IsDirty()) {
        this->colourSlot.ResetDirty();
        float r, g, b;
        if (core::utility::ColourParser::FromString(
                this->colourSlot.Param<core::param::StringParam>()->Value().c_str(), r, g, b)) {
            this->defCol[0] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(r * 255.0f), 0, 255));
            this->defCol[1] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(g * 255.0f), 0, 255));
            this->defCol[2] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(b * 255.0f), 0, 255));
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Unable to parse default colour \"%s\"\n",
                this->colourSlot.Param<core::param::StringParam>()->Value().c_str());
        }
    }
    vislib::Array<int> colMode(this->posData.Count(), this->colourModeSlot.Param<core::param::EnumParam>()->Value(),
        vislib::Array<int>::DEFAULT_CAPACITY_INCREMENT);
    for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
        if ((colMode[i] == 1) && (this->colData[i]->GetSize() == 0)) {
            colMode[i] = 0;
        }
    }
    if (this->dircolourSlot.IsDirty()) {
        this->dircolourSlot.ResetDirty();
        float r, g, b;
        if (core::utility::ColourParser::FromString(
                this->dircolourSlot.Param<core::param::StringParam>()->Value().c_str(), r, g, b)) {
            this->dirdefCol[0] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(r * 255.0f), 0, 255));
            this->dirdefCol[1] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(g * 255.0f), 0, 255));
            this->dirdefCol[2] =
                static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(b * 255.0f), 0, 255));
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Unable to parse default dir::colour \"%s\"\n",
                this->dircolourSlot.Param<core::param::StringParam>()->Value().c_str());
        }
    }
    int dircolMode = this->dircolourModeSlot.Param<core::param::EnumParam>()->Value();

    if (mpdc != NULL) {
        mpdc->SetFrameID(0);
        mpdc->SetDataHash(this->datahash);
        mpdc->SetParticleListCount(static_cast<unsigned int>(this->posData.Count()));
        // TODO hier ne for-schleife um die listen...
        for (int idx = 0; idx < static_cast<int>(this->posData.Count()); idx++) {
            mpdc->AccessParticles(idx).SetGlobalColour(this->defCol[0], this->defCol[1], this->defCol[2]);
            mpdc->AccessParticles(idx).SetGlobalRadius(this->radiusSlot.Param<core::param::FloatParam>()->Value());
            mpdc->AccessParticles(idx).SetCount(this->posData[idx]->GetSize() / (3 * sizeof(float)));
            mpdc->AccessParticles(idx).SetGlobalType(this->typeData[idx]);
            if (this->allDirData[idx]->GetSize() > 0) {
                unsigned int fpp = (dircolMode == 1) ? 7 : ((dircolMode == 2) ? 9 : 6); // floats per particle
                if (this->allDirData[idx]->GetSize() / (fpp * sizeof(float)) != mpdc->AccessParticles(idx).GetCount()) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "IMDAtomDataSource: inconsistent positions and directions, disabling data");
                    mpdc->AccessParticles(idx).SetVertexData(
                        geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE, NULL);
                    mpdc->AccessParticles(idx).SetColourData(
                        geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
                    mpdc->AccessParticles(idx).SetDirData(
                        geocalls::MultiParticleDataCall::Particles::DIRDATA_NONE, NULL);
                } else {
                    mpdc->AccessParticles(idx).SetVertexData(
                        geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                        this->allDirData[idx]->As<void>(), fpp * sizeof(float));
                    if (dircolMode == 1) {
                        if (this->dirautoColumnRangeSlot.Param<core::param::BoolParam>()->Value()) {
                            mpdc->AccessParticles(idx).SetColourMapIndexValues(this->minC[idx], this->maxC[idx]);
                        } else {
                            mpdc->AccessParticles(idx).SetColourMapIndexValues(
                                this->dirminColumnValSlot.Param<core::param::FloatParam>()->Value(),
                                this->dirmaxColumnValSlot.Param<core::param::FloatParam>()->Value());
                        }
                        mpdc->AccessParticles(idx).SetColourData(
                            geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                            this->allDirData[idx]->At(3 * sizeof(float)), fpp * sizeof(float));
                    } else if (dircolMode == 2) {
                        mpdc->AccessParticles(idx).SetColourData(
                            geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB,
                            this->allDirData[idx]->At(3 * sizeof(float)), fpp * sizeof(float));
                    } else {
                        mpdc->AccessParticles(idx).SetColourData(
                            geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
                    }
                    mpdc->AccessParticles(idx).SetDirData(geocalls::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
                        this->allDirData[idx]->At(
                            ((dircolMode == 1) ? 4 : ((dircolMode == 2) ? 6 : 3)) * sizeof(float)),
                        fpp * sizeof(float));
                }
            } else {
                switch (colMode[idx]) {
                case 0:
                    mpdc->AccessParticles(idx).SetColourData(
                        geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
                    break;
                case 1:
                    if (this->autoColumnRangeSlot.Param<core::param::BoolParam>()->Value()) {
                        mpdc->AccessParticles(idx).SetColourMapIndexValues(this->minC[idx], this->maxC[idx]);
                    } else {
                        mpdc->AccessParticles(idx).SetColourMapIndexValues(
                            this->minColumnValSlot.Param<core::param::FloatParam>()->Value(),
                            this->maxColumnValSlot.Param<core::param::FloatParam>()->Value());
                    }
                    mpdc->AccessParticles(idx).SetColourData(
                        geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, this->colData[idx]->As<void>());
                    break;
                default:
                    mpdc->AccessParticles(idx).SetColourData( // some internal error
                        geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
                    break;
                }
            }

            if (!this->posData[idx]->IsEmpty()) {
                mpdc->AccessParticles(idx).SetVertexData(
                    geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, this->posData[idx]->As<void>());
            } else {
                mpdc->AccessParticles(idx).SetVertexData(
                    geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE, NULL);
            }
        }
        mpdc->SetUnlocker(nullptr);
    }
    return true;
}


/*
 * IMDAtomDataSource::getExtentCallback
 */
bool IMDAtomDataSource::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL)
        return false;
    this->assertData();

    float rad = this->radiusSlot.Param<core::param::FloatParam>()->Value();

    if (mpdc != NULL) {
        mpdc->SetDataHash(this->datahash);
        mpdc->SetFrameCount(1);

        if (this->bboxEnabledSlot.Param<core::param::BoolParam>()->Value()) {
            vislib::math::Vector<float, 3> minP(this->bboxMinSlot.Param<core::param::Vector3fParam>()->Value()),
                maxP(this->bboxMaxSlot.Param<core::param::Vector3fParam>()->Value());
            mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(
                minP.GetX(), minP.GetY(), minP.GetZ(), maxP.GetX(), maxP.GetY(), maxP.GetZ());
            mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(
                minP.GetX(), minP.GetY(), minP.GetZ(), maxP.GetX(), maxP.GetY(), maxP.GetZ());
        } else {
            mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->headerMinX, this->headerMinY, this->headerMinZ,
                this->headerMaxX, this->headerMaxY, this->headerMaxZ);
            // this->headerMinX - rad, this->headerMinY - rad, this->headerMinZ - rad,
            // this->headerMaxX + rad, this->headerMaxY + rad, this->headerMaxZ + rad);
            mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->minX - rad, this->minY - rad, this->minZ - rad,
                this->maxX + rad, this->maxY + rad, this->maxZ + rad);
        }
    }

    return true;
}


/*
 * IMDAtomDataSource::clear
 */
void IMDAtomDataSource::clear(void) {
    for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
        this->posData[i]->EnforceSize(0);
        this->colData[i]->EnforceSize(0);
        this->allDirData[i]->EnforceSize(0);
    }
    this->headerMinX = this->headerMinY = this->headerMinZ = 0.0f;
    this->headerMaxX = this->headerMaxY = this->headerMaxZ = 1.0f;
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash++;
}


/*
 * IMDAtomDataSource::assertData
 */
void IMDAtomDataSource::assertData(void) {
    using megamol::core::utility::log::Log;
    if (!this->filenameSlot.IsDirty() && !this->colourModeSlot.IsDirty() && !this->colourColumnSlot.IsDirty() &&
        !this->splitLoadDiredDataSlot.IsDirty() && !this->dirXColNameSlot.IsDirty() &&
        !this->dirYColNameSlot.IsDirty() && !this->dirZColNameSlot.IsDirty() && !this->dircolourModeSlot.IsDirty() &&
        !this->dircolourColumnSlot.IsDirty() && !this->typeColumnSlot.IsDirty() && !this->bboxEnabledSlot.IsDirty() &&
        !this->bboxMaxSlot.IsDirty() && !this->bboxMinSlot.IsDirty())
        return;
    this->filenameSlot.ResetDirty();
    this->colourModeSlot.ResetDirty();
    this->colourColumnSlot.ResetDirty();
    this->splitLoadDiredDataSlot.ResetDirty();
    this->dirXColNameSlot.ResetDirty();
    this->dirYColNameSlot.ResetDirty();
    this->dirZColNameSlot.ResetDirty();
    this->dircolourModeSlot.ResetDirty();
    this->dircolourColumnSlot.ResetDirty();
    this->typeColumnSlot.ResetDirty();
    this->bboxEnabledSlot.ResetDirty();
    this->bboxMaxSlot.ResetDirty();
    this->bboxMinSlot.ResetDirty();

    this->clear();

    vislib::sys::FastFile file;
    auto filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
    //    Log::DefaultLog.WriteInfo(50, _T("Loading \"%s\""), filename.PeekBuffer());
    // this->datahash = static_cast<SIZE_T>(filename.HashCode());
    if (!file.Open(filename.native().c_str(), vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
            vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open imd file %s\n", filename.generic_u8string().c_str());
        return;
    }

    HeaderData header;
    if (!this->readHeader(file, header)) {
        // error already logged
        file.Close();
        return;
    }

    //    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
    //        "IMDAtom with %d data colums:\n", static_cast<int>(header.captions.Count()));
    //    for (SIZE_T i = 0; i < header.captions.Count(); i++) {
    //        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
    //            "\t%s\n", header.captions[i].PeekBuffer());
    //    }

    UINT32 endianTestInt = 0x12345678;
    UINT8 endianTestBytes[4];
    ::memcpy(endianTestBytes, &endianTestInt, 4);
    bool machineLittleEndian = ((endianTestBytes[0] == 0x78) && (endianTestBytes[1] == 0x56) &&
                                (endianTestBytes[2] == 0x34) && (endianTestBytes[3] == 0x12));

    vislib::StringA dirXColName = this->dirXColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    vislib::StringA dirYColName = this->dirYColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    vislib::StringA dirZColName = this->dirZColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    INT_PTR dirXCol = dirXColName.IsEmpty() ? -1 : header.captions.IndexOf(dirXColName);
    INT_PTR dirYCol = dirYColName.IsEmpty() ? -1 : header.captions.IndexOf(dirYColName);
    INT_PTR dirZCol = dirZColName.IsEmpty() ? -1 : header.captions.IndexOf(dirZColName);
    // TODO hier die type column? vermutlich nicht.
    bool loadDir = (dirXCol >= 0) && (dirYCol >= 0) && (dirZCol >= 0);
    bool splitLoadDir = this->splitLoadDiredDataSlot.Param<core::param::BoolParam>()->Value();

    bool retval = false;
    switch (header.format) {
    case 'A': // ASCII
        retval = this->readData<AtomReaderASCII>(file, header, loadDir, splitLoadDir);
        break;
    case 'B': // binary, big endian, double
        retval = (machineLittleEndian) ? this->readData<AtomReaderDoubleSwitched>(file, header, loadDir, splitLoadDir)
                                       : this->readData<AtomReaderDouble>(file, header, loadDir, splitLoadDir);
        break;
    case 'b': // binary, big endian, float
        retval = (machineLittleEndian) ? this->readData<AtomReaderFloatSwitched>(file, header, loadDir, splitLoadDir)
                                       : this->readData<AtomReaderFloat>(file, header, loadDir, splitLoadDir);
        break;
    case 'L': // binary, little endian, double
        retval = (machineLittleEndian) ? this->readData<AtomReaderDouble>(file, header, loadDir, splitLoadDir)
                                       : this->readData<AtomReaderDoubleSwitched>(file, header, loadDir, splitLoadDir);
        break;
    case 'l': // binary, little endian float
        retval = (machineLittleEndian) ? this->readData<AtomReaderFloat>(file, header, loadDir, splitLoadDir)
                                       : this->readData<AtomReaderFloatSwitched>(file, header, loadDir, splitLoadDir);
        break;
    default:
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read imd file: Illegal format\n");
        break;
    }

    if (retval) {
        // TODO inside readData!
        // this->posData.EnforceSize(posWriter.End(), true);
        // this->colData.EnforceSize(colWriter.End(), true);
        // this->allDirData.EnforceSize(dirWriter.End(), true);
        // TODO allTypeData

        SIZE_T cnt = 0;
        for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
            if (this->allDirData[i]->GetSize() > 0) {
                unsigned int dircolMode = this->dircolourModeSlot.Param<core::param::EnumParam>()->Value();
                unsigned int fpp = (dircolMode == 1) ? 7 : ((dircolMode == 2) ? 9 : 6); // floats per particle
                cnt += this->allDirData[i]->GetSize() / (fpp * sizeof(float));
            } else {
                cnt += this->posData[i]->GetSize() / (sizeof(float) * 3);
            }
        }
        //        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%d Atoms loaded\n", cnt);
        //        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Data bounding box = (%f, %f, %f) ... (%f, %f, %f)\n",
        //            this->minX, this->minY, this->minZ, this->maxX, this->maxY, this->maxZ);

        // this->datahash = (this->datahash << (sizeof(SIZE_T) / 2))
        //    || (this->datahash >> (sizeof(SIZE_T) / 2));
        // this->datahash ^= this->posData.GetSize();
        this->datahash++;

        // All parameters must influence the data hash

    } else {
        // error already logged
        // this->posData.EnforceSize(0, true);
        // this->colData.EnforceSize(0, true);
        // this->allDirData.EnforceSize(0, true);
        // this->datahash = 0;
    }

    file.Close();

    // apply filter (if activated)
    this->posXFilterUpdate(this->posXFilterNow);
}


/*
 * IMDAtomDataSource::readHeader
 */
bool IMDAtomDataSource::readHeader(vislib::sys::File& file, IMDAtomDataSource::HeaderData& header) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    vislib::StringA line;
    bool windowsNewline = false;
    char nlb[2];

    try {
        file.SeekToBegin();
        line = vislib::sys::ReadLineFromFileA(file);
        if (line[0] != '#') {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Failed to parse IMD header: Illegal first line character %c\n", line[0]);
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
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_WARN, "Line %d has illegal first character: %c\n", linePos, line[0]);
                warnCnt++;
                if (warnCnt == 10) {
                    throw new vislib::Exception("Too many warnings", __FILE__, __LINE__);
                } else {
                    continue;
                }
            }

            switch (line[1]) {
            case '#':
                break;  // comment line
            case 'F': { // format line
                line.Replace('\t', ' ');
                vislib::Array<vislib::StringA> itemz = vislib::StringTokeniserA::Split(line, ' ', true);
                if (itemz.Count() != 8) {
                    vislib::StringA msg("Illegal format line (not 7 fields): ");
                    msg.Append(line);
                    throw new vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                }

                header.format = itemz[1][0];
                if (((header.format != 'A') && (header.format != 'B') && (header.format != 'b') &&
                        (header.format != 'L') && (header.format != 'l')) ||
                    (itemz[1].Length() != 1)) {
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
                vislib::Array<vislib::StringA> itemz = vislib::StringTokeniserA::Split(line, ' ', true);
                if (itemz.Count() < 3) {
                    throw new vislib::Exception("Illegal bounding box x vector", __FILE__, __LINE__);
                }
                vislib::math::Vector<double, 3> vec(
                    vislib::CharTraitsA::ParseDouble(itemz[1]), vislib::CharTraitsA::ParseDouble(itemz[2]), 0.0);
                if (itemz.Count() >= 4) {
                    vec.SetZ(vislib::CharTraitsA::ParseDouble(itemz[3]));
                }
                this->headerMaxX = static_cast<float>(vec.Length());
            } break;
            case 'Y': { // bounding box y line
                line.Replace('\t', ' ');
                vislib::Array<vislib::StringA> itemz = vislib::StringTokeniserA::Split(line, ' ', true);
                if (itemz.Count() < 3) {
                    throw new vislib::Exception("Illegal bounding box y vector", __FILE__, __LINE__);
                }
                vislib::math::Vector<double, 3> vec(
                    vislib::CharTraitsA::ParseDouble(itemz[1]), vislib::CharTraitsA::ParseDouble(itemz[2]), 0.0);
                if (itemz.Count() >= 4) {
                    vec.SetZ(vislib::CharTraitsA::ParseDouble(itemz[3]));
                }
                this->headerMaxY = static_cast<float>(vec.Length());
            } break;
            case 'Z': { // bounding box z line
                line.Replace('\t', ' ');
                vislib::Array<vislib::StringA> itemz = vislib::StringTokeniserA::Split(line, ' ', true);
                if (itemz.Count() < 4) {
                    throw new vislib::Exception("Illegal bounding box z vector", __FILE__, __LINE__);
                }
                vislib::math::Vector<double, 3> vec(vislib::CharTraitsA::ParseDouble(itemz[1]),
                    vislib::CharTraitsA::ParseDouble(itemz[2]), vislib::CharTraitsA::ParseDouble(itemz[3]));
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
                    if (header.id)
                        cnt++;
                    if (header.type)
                        cnt++;
                    if (header.mass)
                        cnt++;
                    cnt += header.pos + header.vel + header.dat;
                    int hcnt = static_cast<int>(header.captions.Count());
                    if (hcnt < cnt) {
                        Log::DefaultLog.WriteMsg(
                            Log::LEVEL_WARN, "Too few data column captions specified (%d instead of %d)", hcnt, cnt);
                        for (; hcnt < cnt; hcnt++) {
                            header.captions.Add("unnamed");
                        }
                    } else if (hcnt > cnt) {
                        Log::DefaultLog.WriteMsg(
                            Log::LEVEL_WARN, "Too many data column captions specified (%d instead of %d)", hcnt, cnt);
                        header.captions.Erase(cnt, hcnt - cnt);
                    }
                }

                return true;
            }
        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to parse IMD header: unexpected end of file\n");

    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to parse IMD header: %s\n", ex.GetMsgA());
    } catch (...) { Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to parse IMD header: unexpected exception\n"); }

    return false;
}

template<typename T>
void IMDAtomDataSource::readToIntColumn(T& reader, bool& fail, unsigned int* column, ...) {
    const int numKnownColumns = 6;
    float* out;
    unsigned int col;
    va_list vl;
    va_start(vl, column);
    for (int i = 0; i < numKnownColumns; i++) {
        col = va_arg(vl, unsigned int);
        out = va_arg(vl, float*);
        if (col == *column) {
            *out = static_cast<float>(reader.ReadInt(fail));
            va_end(vl);
            (*column)++;
            return;
        }
    }
    va_end(vl);
    reader.SkipInt(fail);
    (*column)++;
}

template<typename T>
void IMDAtomDataSource::readToFloatColumn(T& reader, bool& fail, unsigned int* column, ...) {
    const int numKnownColumns = 6;
    float* out;
    unsigned int col;
    va_list vl;
    va_start(vl, column);
    for (int i = 0; i < numKnownColumns; i++) {
        col = va_arg(vl, unsigned int);
        out = va_arg(vl, float*);
        if (col == *column) {
            *out = static_cast<float>(reader.ReadFloat(fail));
            va_end(vl);
            (*column)++;
            return;
        }
    }
    va_end(vl);
    reader.SkipFloat(fail);
    (*column)++;
}

/*
 * IMDAtomDataSource::readData
 */
template<typename T>
bool IMDAtomDataSource::readData(
    vislib::sys::File& file, const IMDAtomDataSource::HeaderData& header, bool loadDir, bool splitDir) {
    T reader(file);
    bool fail = false;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    bool first = true;
    float c = 0.0f, dc = 0.0f, t = 0.0f;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    unsigned int column;
    unsigned int colcolumn = UINT_MAX;
    unsigned int dircolcolumn = UINT_MAX;
    unsigned int typecolumn = UINT_MAX;
    // TODO type column

    this->typeData.Clear();
    this->minC.Clear();
    this->maxC.Clear();
    this->posData.Clear();
    this->colData.Clear();
    this->allDirData.Clear();

    vislib::PtrArray<vislib::RawStorageWriter> posWriters;
    vislib::PtrArray<vislib::RawStorageWriter> colWriters;
    vislib::PtrArray<vislib::RawStorageWriter> dirWriters;

    bool normaliseDir = this->dirNormDirSlot.Param<core::param::BoolParam>()->Value();
    vislib::StringA dirXColName = this->dirXColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    vislib::StringA dirYColName = this->dirYColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    vislib::StringA dirZColName = this->dirZColNameSlot.Param<core::param::StringParam>()->Value().c_str();
    INT_PTR dirXCol = dirXColName.IsEmpty() ? -1 : header.captions.IndexOf(dirXColName);
    INT_PTR dirYCol = dirYColName.IsEmpty() ? -1 : header.captions.IndexOf(dirYColName);
    INT_PTR dirZCol = dirZColName.IsEmpty() ? -1 : header.captions.IndexOf(dirZColName);
    ASSERT(!loadDir || (dirXCol >= 0));
    ASSERT(!loadDir || (dirYCol >= 0));
    ASSERT(!loadDir || (dirZCol >= 0));
    int dircolMode = this->dircolourModeSlot.Param<core::param::EnumParam>()->Value();

    if (true) {
        // type from column
        vislib::StringA typecolname(this->typeColumnSlot.Param<core::param::StringParam>()->Value().c_str());

        // 1. exact match
        for (SIZE_T i = 0; i < header.captions.Count(); i++) {
            if (header.captions[i].Equals(typecolname)) {
                typecolumn = static_cast<unsigned int>(i);
                break;
            }
        }

        if (typecolumn == UINT_MAX) {
            // 2. caseless match
            for (SIZE_T i = 0; i < header.captions.Count(); i++) {
                if (header.captions[i].Equals(typecolname, false)) {
                    typecolumn = static_cast<unsigned int>(i);
                    break;
                }
            }
        }

        if (typecolumn == UINT_MAX) {
            // 3. index
            try {
                typecolumn = vislib::CharTraitsA::ParseInt(typecolname);
                if (typecolumn >= static_cast<unsigned int>(header.captions.Count())) {
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                        "The parsed type column index is out of range (%u not in 0..%d)\n", typecolumn,
                        static_cast<int>(header.captions.Count()) - 1);
                    typecolumn = UINT_MAX;
                }
            } catch (...) { typecolumn = UINT_MAX; }
        }

        if (typecolumn == UINT_MAX) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Failed to parse type column selection: %s\n", typecolname.PeekBuffer());
        }
    }

    if (this->colourModeSlot.Param<core::param::EnumParam>()->Value() == 1) {
        // column colouring mode
        vislib::StringA colcolname(this->colourColumnSlot.Param<core::param::StringParam>()->Value().c_str());

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
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                        "The parsed colouring column index is out of range (%u not in 0..%d)\n", colcolumn,
                        static_cast<int>(header.captions.Count()) - 1);
                    colcolumn = UINT_MAX;
                }
            } catch (...) { colcolumn = UINT_MAX; }
        }

        if (colcolumn == UINT_MAX) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Failed to parse colour column selection: %s\n", colcolname.PeekBuffer());
        }
    }

    if (dircolMode == 1) {
        // column colouring mode
        vislib::StringA dircolcolname(this->dircolourColumnSlot.Param<core::param::StringParam>()->Value().c_str());
        // 1. exact match
        for (SIZE_T i = 0; i < header.captions.Count(); i++) {
            if (header.captions[i].Equals(dircolcolname)) {
                dircolcolumn = static_cast<unsigned int>(i);
                break;
            }
        }
        if (dircolcolumn == UINT_MAX) {
            // 2. caseless match
            for (SIZE_T i = 0; i < header.captions.Count(); i++) {
                if (header.captions[i].Equals(dircolcolname, false)) {
                    dircolcolumn = static_cast<unsigned int>(i);
                    break;
                }
            }
        }
        if (dircolcolumn == UINT_MAX) {
            // 3. index
            try {
                dircolcolumn = vislib::CharTraitsA::ParseInt(dircolcolname);
                if (dircolcolumn >= static_cast<unsigned int>(header.captions.Count())) {
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                        "The parsed dir colouring column index is out of range (%u not in 0..%d)\n", dircolcolumn,
                        static_cast<int>(header.captions.Count()) - 1);
                    dircolcolumn = UINT_MAX;
                }
            } catch (...) { dircolcolumn = UINT_MAX; }
        }
        if (dircolcolumn == UINT_MAX) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Failed to parse dir colour column selection: %s\n", dircolcolname.PeekBuffer());
        }
    }

    while (!fail) {
        column = 0;

        if (header.id) {
            // these columns can be filled from anything you put into the respective parameters (by column name)
            // problem: type does not exist!
            // TODO  refactor, add type column
            this->readToIntColumn(reader, fail, &column, colcolumn, &c, dirXCol, &dx, dirYCol, &dy, dirZCol, &dz,
                dircolcolumn, &dc, typecolumn, &t);
            // if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) || (column
            // == dircolcolumn)){
            //    float f = static_cast<float>(reader.ReadInt(fail));
            //    if (column == colcolumn) c = f;
            //    if (column == dirXCol) dx = f;
            //    if (column == dirYCol) dy = f;
            //    if (column == dirZCol) dz = f;
            //    if (column == dircolcolumn) dc = f;
            //} else {
            //    reader.SkipInt(fail);
            //}
            // column++;
        }
        if (header.type) {
            this->readToIntColumn(reader, fail, &column, colcolumn, &c, dirXCol, &dx, dirYCol, &dy, dirZCol, &dz,
                dircolcolumn, &dc, typecolumn, &t);
            // if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) || (column
            // == dircolcolumn)){
            //    float f = static_cast<float>(reader.ReadInt(fail));
            //    if (column == colcolumn) c = f;
            //    if (column == dirXCol) dx = f;
            //    if (column == dirYCol) dy = f;
            //    if (column == dirZCol) dz = f;
            //    if (column == dircolcolumn) dc = f;
            //} else {
            //    reader.SkipInt(fail);
            //}
            // column++;
        }
        if (header.mass) {
            this->readToFloatColumn(reader, fail, &column, colcolumn, &c, dirXCol, &dx, dirYCol, &dy, dirZCol, &dz,
                dircolcolumn, &dc, typecolumn, &t);
            // if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) || (column
            // == dircolcolumn)){
            //    float f = reader.ReadFloat(fail);
            //    if (column == colcolumn) c = f;
            //    if (column == dirXCol) dx = f;
            //    if (column == dirYCol) dy = f;
            //    if (column == dirZCol) dz = f;
            //    if (column == dircolcolumn) dc = f;
            //} else {
            //    reader.SkipFloat(fail);
            //}
            // column++;
        }
        for (int i = 0; i < header.pos; i++) {
            if (i == 0) {
                x = reader.ReadFloat(fail);
                if (column == colcolumn)
                    c = x;
                if (column == dirXCol)
                    dx = x;
                if (column == dirYCol)
                    dy = x;
                if (column == dirZCol)
                    dz = x;
                if (column == dircolcolumn)
                    dc = x;
            }
            if (i == 1) {
                y = reader.ReadFloat(fail);
                if (column == colcolumn)
                    c = y;
                if (column == dirXCol)
                    dx = y;
                if (column == dirYCol)
                    dy = y;
                if (column == dirZCol)
                    dz = y;
                if (column == dircolcolumn)
                    dc = y;
            }
            if (i == 2) {
                z = reader.ReadFloat(fail);
                if (column == colcolumn)
                    c = z;
                if (column == dirXCol)
                    dx = z;
                if (column == dirYCol)
                    dy = z;
                if (column == dirZCol)
                    dz = z;
                if (column == dircolcolumn)
                    dc = z;
            }
            if (i >= 3) {
                if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) ||
                    (column == dircolcolumn)) {
                    float f = reader.ReadFloat(fail);
                    if (column == colcolumn)
                        c = f;
                    if (column == dirXCol)
                        dx = f;
                    if (column == dirYCol)
                        dy = f;
                    if (column == dirZCol)
                        dz = f;
                    if (column == dircolcolumn)
                        dc = f;
                } else {
                    reader.SkipFloat(fail);
                }
            }
            column++;
        }
        for (int i = 0; i < header.vel; i++) {
            this->readToFloatColumn(reader, fail, &column, colcolumn, &c, dirXCol, &dx, dirYCol, &dy, dirZCol, &dz,
                dircolcolumn, &dc, typecolumn, &t);
            // if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) || (column
            // == dircolcolumn)){
            //    float f = reader.ReadFloat(fail);
            //    if (column == colcolumn) c = f;
            //    if (column == dirXCol) dx = f;
            //    if (column == dirYCol) dy = f;
            //    if (column == dirZCol) dz = f;
            //    if (column == dircolcolumn) dc = f;
            //} else {
            //    reader.SkipFloat(fail);
            //}
            // column++;
        }
        for (int i = 0; i < header.dat; i++) {
            this->readToFloatColumn(reader, fail, &column, colcolumn, &c, dirXCol, &dx, dirYCol, &dy, dirZCol, &dz,
                dircolcolumn, &dc, typecolumn, &t);
            // if ((column == colcolumn) || (column == dirXCol) || (column == dirYCol) || (column == dirZCol) || (column
            // == dircolcolumn)){
            //    float f = reader.ReadFloat(fail);
            //    if (column == colcolumn) c = f;
            //    if (column == dirXCol) dx = f;
            //    if (column == dirYCol) dy = f;
            //    if (column == dirZCol) dz = f;
            //    if (column == dircolcolumn) dc = f;
            //} else {
            //    reader.SkipFloat(fail);
            //}
            // column++;
        }

        if (!fail) {

            if (this->bboxEnabledSlot.Param<core::param::BoolParam>()->Value()) {
                vislib::math::Vector<float, 3> p(x, y, z),
                    minP(this->bboxMinSlot.Param<core::param::Vector3fParam>()->Value()),
                    maxP(this->bboxMaxSlot.Param<core::param::Vector3fParam>()->Value());
                if ((p.GetX() < minP.GetX() || p.GetY() < minP.GetY() || p.GetZ() < minP.GetZ()) ||
                    (p.GetX() > maxP.GetX() || p.GetY() > maxP.GetY() || p.GetZ() > maxP.GetZ()))
                    continue;
            }


            int rawIdx = 0;
            if ((rawIdx = static_cast<int>(typeData.IndexOf(static_cast<unsigned int>(t)))) ==
                static_cast<int>(vislib::Array<unsigned int>::INVALID_POS)) {
                typeData.Append(static_cast<unsigned int>(t));
                rawIdx = static_cast<int>(typeData.Count() - 1);
                this->posData.Append(new vislib::RawStorage());
                this->colData.Append(new vislib::RawStorage());
                this->allDirData.Append(new vislib::RawStorage());
                this->minC.Append(0.0f);
                this->maxC.Append(1.0f);
                posWriters.Append(new vislib::RawStorageWriter(*(this->posData[rawIdx]), 0, 0, 10 * 1024 * 1024));
                colWriters.Append(new vislib::RawStorageWriter(*(this->colData[rawIdx]), 0, 0, 10 * 1024 * 1024));
                dirWriters.Append(new vislib::RawStorageWriter(*(this->allDirData[rawIdx]), 0, 0, 10 * 1024 * 1024));
            }

            if (!first) {
                if (this->minX > x)
                    this->minX = x;
                else if (this->maxX < x)
                    this->maxX = x;
                if (this->minY > y)
                    this->minY = y;
                else if (this->maxY < y)
                    this->maxY = y;
                if (this->minZ > z)
                    this->minZ = z;
                else if (this->maxZ < z)
                    this->maxZ = z;
            } else {
                first = false;
                this->minX = this->maxX = x;
                this->minY = this->maxY = y;
                this->minZ = this->maxZ = z;
                this->minC[rawIdx] = this->maxC[rawIdx] = c;
            }
            if (colcolumn != UINT_MAX) {
                if (this->minC[rawIdx] > c)
                    this->minC[rawIdx] = c;
                else if (this->maxC[rawIdx] < c)
                    this->maxC[rawIdx] = c;
            }
            if (dircolcolumn != UINT_MAX) {
                if (this->minC[rawIdx] > dc)
                    this->minC[rawIdx] = dc;
                else if (this->maxC[rawIdx] < dc)
                    this->maxC[rawIdx] = dc;
            }

            if (loadDir) {
                if (normaliseDir) {
                    vislib::math::Vector<float, 3> dv(dx, dy, dz);
                    dv.Normalise();
                    dx = dv.X();
                    dy = dv.Y();
                    dz = dv.Z();
                }
                if (splitDir && vislib::math::IsEqual(dx, 0.0f) && vislib::math::IsEqual(dy, 0.0f) &&
                    vislib::math::IsEqual(dz, 0.0f)) {
                    *posWriters[rawIdx] << x << y << z;
                    if (colcolumn != UINT_MAX)
                        *colWriters[rawIdx] << c;
                    // TODO type column??? vermutlich net
                } else {
                    *dirWriters[rawIdx] << x << y << z;
                    if (dircolMode == 2) {
                        vislib::math::Vector<float, 3> dv(dx, dy, dz);
                        dv.Normalise();
                        float xr = 1.0f, xg = 0.0f, xb = 0.0f, yr = 0.0f, yg = 1.0f, yb = 0.0f, zr = 0.0f, zg = 0.0f,
                              zb = 1.0f;
                        if (dv.X() < 0.0f) {
                            xr = 1.0f - xr;
                            xg = 1.0f - xg;
                            xb = 1.0f - xb;
                        }
                        if (dv.Y() < 0.0f) {
                            yr = 1.0f - yr;
                            yg = 1.0f - yg;
                            yb = 1.0f - yb;
                        }
                        if (dv.Z() < 0.0f) {
                            zr = 1.0f - zr;
                            zg = 1.0f - zg;
                            zb = 1.0f - zb;
                        }
                        dv.Set(dv.X() * dv.X(), dv.Y() * dv.Y(), dv.Z() * dv.Z());

                        *dirWriters[rawIdx] << (xr * dv.X() + yr * dv.Y() + zr * dv.Z());
                        *dirWriters[rawIdx] << (xg * dv.X() + yg * dv.Y() + zg * dv.Z());
                        *dirWriters[rawIdx] << (xb * dv.X() + yb * dv.Y() + zb * dv.Z());

                    } else if (dircolcolumn != UINT_MAX)
                        *dirWriters[rawIdx] << dc;
                    else if (colcolumn != UINT_MAX)
                        *dirWriters[rawIdx] << c;
                    *dirWriters[rawIdx] << dx << dy << dz;
                }
            } else {
                *posWriters[rawIdx] << x << y << z;
                if (colcolumn != UINT_MAX)
                    *colWriters[rawIdx] << c;
            }
        }
    }

    if (!first) {
        for (int i = 0; i < static_cast<int>(posData.Count()); i++) {
            this->posData[i]->EnforceSize(posWriters[i]->End(), true);
            this->colData[i]->EnforceSize(colWriters[i]->End(), true);
            this->allDirData[i]->EnforceSize(dirWriters[i]->End(), true);
        }
    } else {
        for (int i = 0; i < static_cast<int>(posData.Count()); i++) {
            this->posData[i]->EnforceSize(0, true);
            this->colData[i]->EnforceSize(0, true);
            this->allDirData[i]->EnforceSize(0, true);
        }
    }
    return !first;
}

// TODO das ist eigentlich kruscht, das sollte wenn dann ein region-filter sein, aber na gut...
/*
 * IMDAtomDataSource::posXFilterUpdate
 */
bool IMDAtomDataSource::posXFilterUpdate(core::param::ParamSlot& slot) {
    using megamol::core::utility::log::Log;
    if (!this->posXFilter.Param<core::param::BoolParam>()->Value()) {
        //        Log::DefaultLog.WriteInfo("PosX-Filter not enabled");
        return true;
    }

    float minX = this->posXMinFilter.Param<core::param::FloatParam>()->Value();
    float maxX = this->posXMaxFilter.Param<core::param::FloatParam>()->Value();

    // number of input particles
    for (int idx = 0; idx < static_cast<int>(this->posData.Count()); idx++) {
        SIZE_T inCnt = this->posData[idx]->GetSize() / (3 * sizeof(float));
        SIZE_T outCnt = 0;
        float* outPos = this->posData[idx]->As<float>();
        for (SIZE_T i = 0; i < inCnt; i++) {
            float* inPos = this->posData[idx]->AsAt<float>(i * 3 * sizeof(float));
            if ((inPos[0] >= minX) && (inPos[0] <= maxX)) {
                if (inPos != outPos) {
                    ::memcpy(outPos, inPos, 3 * sizeof(float));
                }
                outCnt++;
                outPos += 3;
            }
        }

        if (outCnt != inCnt) {
            this->datahash++;
            this->posData[idx]->EnforceSize(outCnt * 3 * sizeof(float), true);

            Log::DefaultLog.WriteInfo("PosX-Filtered from %u to %u particles\n", static_cast<unsigned int>(inCnt),
                static_cast<unsigned int>(outCnt));
        } else {
            Log::DefaultLog.WriteInfo("PosX-Filter did not apply");
        }
    }
    return true;
}
