/*
 * BezDatReader.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "BezDatReader.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/VersionNumber.h"
#include "vislib/sys/Log.h"
#include "vislib/String.h"
#include "vislib/CharTraits.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"

using namespace megamol;
using namespace megamol::beztube;


/*
 * BezDatReader::FileFormatAutoDetect
 */
float BezDatReader::FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize) {
    if (dataSize < 7) return 0.0f; // insufficient data
    if (::memcmp(data, "BezDat", 6) != 0) return 0.0f; // not a BezDat
    char tc = reinterpret_cast<const char*>(data)[7];
    if ((tc != 'A') && (tc != 'B')) return 0.0f; // neighter ASCII nor binary

    if (tc == 'A') {
        SIZE_T len = 7;
        while ((len < dataSize) && (reinterpret_cast<const char*>(data)[len] != '\n')) len++;
        if (len >= dataSize) return 0.5f; // header start ok, but insufficient data to confirm

        vislib::StringA line(reinterpret_cast<const char*>(data), static_cast<vislib::StringA::Size>(len));
        line = line.Substring(7);
        line.TrimSpaces();

        if (line.IsEmpty()) return 0.125f; // header start looked ok, but failed to find the version number

        try {
            vislib::VersionNumber vn;
            vn.Parse(line.PeekBuffer());

            if ((vn == vislib::VersionNumber(1, 0))
                    || (vn == vislib::VersionNumber(2, 0))) {
                return 1.0f; // all looks great
            }

        } catch(...) {
            // failed to parse version number
        }

    } else if (tc == 'B') {
        if (dataSize < 24) return 0.5f; // header start ok, but insufficient data to confirm

        if ((data[12] != 2) || (data[13] != 0) || (data[14] != 0) || (data[15] != 0)) return 0.0f; // version number wrong

        unsigned int et1;
        float et2;
        ::memcpy(&et1, data + 16, 4);
        ::memcpy(&et2, data + 20, 4);

        if ((et1 != 0x12345678) || !vislib::math::IsEqual(et2, 3.141f)) return 0.0f; // unsupported endian

        return 1.0f; // all looks great
    }

    return 0.0f;
}


/*
 * BezDatReader::BezDatReader
 */
BezDatReader::BezDatReader(void) : Module(),
        outDataSlot("outData", "Slot providing data"),
        filenameSlot("filename", "Slot for the file name to load"),
        dataHash(0), data(), hasStaticIndices(false) {

    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetData", &BezDatReader::getData);
    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetExtent", &BezDatReader::getExtent);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);
}


/*
 * BezDatReader::~BezDatReader
 */
BezDatReader::~BezDatReader(void) {
    this->Release();
}


/*
 * BezDatReader::create
 */
bool BezDatReader::create(void) {
    // intentionally empty
    return true;
}


/*
 * BezDatReader::release
 */
void BezDatReader::release(void) {
    // intentionally empty
}


/*
 * BezDatReader::getData
 */
bool BezDatReader::getData(megamol::core::Call& call) {
    core::misc::BezierCurvesListDataCall *bcldc
        = dynamic_cast<core::misc::BezierCurvesListDataCall*>(&call);
    if (bcldc == NULL) return false;
    this->assertData();

    if (this->data.Count() <= 0) return false; // no data

    unsigned int fidx = bcldc->FrameID();
    if (fidx >= this->data.Count()) {
        fidx = static_cast<unsigned int>(this->data.Count() - 1);
        bcldc->SetFrameID(fidx);
    }

    bcldc->SetData(
        this->data[fidx].First().PeekElements(),
        this->data[fidx].First().Count(),
        this->hasStaticIndices);

    bcldc->SetDataHash(this->dataHash);
    bcldc->SetUnlocker(nullptr);

    return true;
}


/*
 * BezDatReader::getExtent
 */
bool BezDatReader::getExtent(megamol::core::Call& call) {
    core::misc::BezierCurvesListDataCall *bcldc
        = dynamic_cast<core::misc::BezierCurvesListDataCall*>(&call);
    if (bcldc == NULL) return false;
    this->assertData();

    if (this->data.Count() <= 0) return false; // no data

    unsigned int fidx = bcldc->FrameID();
    if (fidx >= this->data.Count()) fidx = 0;

    bcldc->SetExtent(
        static_cast<unsigned int>(this->data.Count()),
        this->data[fidx].Second());
    bcldc->SetDataHash(this->dataHash);
    bcldc->SetHasStaticIndices(this->hasStaticIndices);
    bcldc->SetUnlocker(nullptr);

    return true;
}


/*
 * BezDatReader::assertData
 */
void BezDatReader::assertData(void) {
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->loadData(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
        this->dataHash++;
    }
}


/*
 * BezDatReader::loadData
 */
void BezDatReader::loadData(const vislib::TString& filename) {
    using vislib::sys::Log;

    this->clear();

    vislib::sys::File file;
    if (!file.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteError("Failed to open \"%s\"", vislib::StringA(filename).PeekBuffer());
        return;
    }

    char h1[7];
    if (file.Read(h1, 7) != 7) {
        Log::DefaultLog.WriteError("Failed to read header start of \"%s\"", vislib::StringA(filename).PeekBuffer());
        file.Close();
        return;
    }

    if (::memcmp(h1, "BezDat", 6) != 0) {
        Log::DefaultLog.WriteError("File header of \"%s\" invalid", vislib::StringA(filename).PeekBuffer());
        file.Close();
        return;
    }

    try {
        if (h1[6] == 'A') {
            // continue with ASCII
            Log::DefaultLog.WriteInfo("Loading \"%s\" as BezDat ASCII", vislib::StringA(filename).PeekBuffer());
            file.SeekToBegin();
            vislib::sys::ASCIIFileBuffer filedata(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
            filedata.LoadFile(file);
            file.Close();

            if ((filedata.Count() < 1) || (filedata[0].Count() < 2)) {
                throw vislib::Exception("File seems truncated", __FILE__, __LINE__);
            }

            vislib::VersionNumber ver(0, 0, 0);
            try {
                ver.Parse(filedata[0].Word(1));
            } catch(...) {
            }
            if (ver == vislib::VersionNumber(1, 0)) {
                this->loadASCII_1_0(filedata);

            } else if (ver == vislib::VersionNumber(2, 0)) {
                this->loadASCII_2_0(filedata);

            } else {
                throw vislib::Exception("File format version is unsupported", __FILE__, __LINE__);
            }

        } else if (h1[6] == 'B') {
            // continue with binary
            Log::DefaultLog.WriteInfo("Loading \"%s\" as BezDat binary", vislib::StringA(filename).PeekBuffer());

            file.Seek(5, vislib::sys::File::CURRENT);
            if (file.Read(h1, 4) != 4) {
                throw vislib::Exception("File seems truncated", __FILE__, __LINE__);
            }

            if ((h1[0] != 2) || (h1[1] != 0) || (h1[2] != 0) || (h1[3] != 0)) {
                throw vislib::Exception("File format version is unsupported", __FILE__, __LINE__);
            }

            unsigned int et1;
            float et2;
            if (file.Read(&et1, 4) != 4) {
                throw vislib::Exception("File seems truncated", __FILE__, __LINE__);
            }
            if (file.Read(&et2, 4) != 4) {
                throw vislib::Exception("File seems truncated", __FILE__, __LINE__);
            }

            if ((et1 != 0x12345678) || !vislib::math::IsEqual(et2, 3.141f)) {
                throw vislib::Exception("File stored with incompatible endianess", __FILE__, __LINE__);
            }

            this->loadBinary_2_0(file);

        } else {
            Log::DefaultLog.WriteError("File format type of \"%s\" invalid", vislib::StringA(filename).PeekBuffer());
        }
    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("%s", ex.GetMsgA());
    } catch(...) {
        Log::DefaultLog.WriteError("Failed to load due to an unexpected exception");
    }
    file.Close();
}


/*
 * BezDatReader::clear
 */
void BezDatReader::clear(void) {
    this->data.Clear();
    this->hasStaticIndices = false;
}


/*
 * BezDatReader::loadBinary_2_0
 */
void BezDatReader::loadBinary_2_0(vislib::sys::File& file) {
    using core::misc::BezierCurvesListDataCall;
    unsigned int frameCnt;
#define MY_FILE_READ(A, B) if (file.Read(A, B) != B) throw vislib::Exception("File seems truncated", __FILE__, __LINE__);
#define MY_CHECKED_FILE_READ(A, B) if (B > frameSize) throw vislib::Exception("File seems corrupted", __FILE__, __LINE__); \
    else { MY_FILE_READ(A, B); frameSize -= B; }

    MY_FILE_READ(&frameCnt, 4);
    if (frameCnt == 0) throw vislib::Exception("File seems corrupted", __FILE__, __LINE__);

    this->data.AssertCapacity(frameCnt);

    for (unsigned int frameIdx = 0; frameIdx < frameCnt; frameIdx++) {
        this->data.Add(vislib::Pair<vislib::Array<core::misc::BezierCurvesListDataCall::Curves>,
            core::BoundingBoxes>());
        vislib::Pair<vislib::Array<core::misc::BezierCurvesListDataCall::Curves>,
            core::BoundingBoxes>& frame = this->data[frameIdx];

        unsigned char *ptData = nullptr;
        unsigned int *idxData = nullptr;

        try {
            unsigned int frameSize = 0;
            MY_FILE_READ(&frameSize, 4);
            if (frameSize < 4) throw vislib::Exception("File seems corrupted", __FILE__, __LINE__);
            frameSize -= 4; // size of frame size itself
            while (frameSize > 0) {
                frame.First().Add(BezierCurvesListDataCall::Curves());
                BezierCurvesListDataCall::Curves& curves = frame.First().Last();

                unsigned char layoutCode = 0;
                MY_CHECKED_FILE_READ(&layoutCode, 1);

                bool hasBBox = (layoutCode & 128) == 128;
                bool hasCBox = (layoutCode & 64) == 64;
                this->hasStaticIndices |= (layoutCode & 32) == 32;

                unsigned int bpp = 0;
                bool needGlobRad = false;
                bool needGlobCol = false;

                switch ((layoutCode & 31)) {
                case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 12; needGlobRad = true; needGlobCol = true; break;
                case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 16; needGlobCol = true; break;
                case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 15; needGlobRad = true; break;
                case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 19; break;
                default: throw vislib::Exception("File seems corrupted", __FILE__, __LINE__);
                }

                if (hasBBox) {
                    float f[6];
                    MY_CHECKED_FILE_READ(f, 4 * 6);
                    frame.Second().SetObjectSpaceBBox(f[0], f[1], f[2], f[3], f[4], f[5]);
                }
                if (hasCBox) {
                    float f[6];
                    MY_CHECKED_FILE_READ(f, 4 * 6);
                    frame.Second().SetObjectSpaceClipBox(f[0], f[1], f[2], f[3], f[4], f[5]);
                }

                if (needGlobRad) {
                    float f;
                    MY_CHECKED_FILE_READ(&f, 4);
                    curves.SetGlobalRadius(f);
                }
                if (needGlobCol) {
                    unsigned char c[3];
                    MY_CHECKED_FILE_READ(c, 3);
                    curves.SetGlobalColour(c[0], c[1], c[2]);
                }

                unsigned int pCnt;
                MY_CHECKED_FILE_READ(&pCnt, 4);
                ptData = new unsigned char[pCnt * bpp];
                MY_CHECKED_FILE_READ(ptData, pCnt * bpp);

                unsigned int iCnt;
                MY_CHECKED_FILE_READ(&iCnt, 4);
                idxData = new unsigned int[iCnt * 4];
                MY_CHECKED_FILE_READ(idxData, iCnt * 4);

                curves.Set(
                    static_cast<BezierCurvesListDataCall::DataLayout>(layoutCode & 31),
                    ptData, pCnt, true, idxData, iCnt, true,
                    curves.GetGlobalRadius(),
                    curves.GetGlobalColour()[0],
                    curves.GetGlobalColour()[1],
                    curves.GetGlobalColour()[2]);

                ptData = nullptr; // Do not delete
                idxData = nullptr; // Do not delete
            }

        } catch(...) {
            if (ptData != nullptr) delete[] ptData;
            if (idxData != nullptr) delete[] idxData;
            this->data.SetCount(this->data.Count() - 1); // last frame was broken
            throw;
        }
    }
#undef MY_FILE_READ
#undef MY_CHECKED_FILE_READ
}


/*
 * BezDatReader::loadASCII_1_0
 */
void BezDatReader::loadASCII_1_0(vislib::sys::ASCIIFileBuffer& file) {
    using vislib::StringA;
    using vislib::sys::Log;
    using vislib::CharTraitsA;
    this->data.SetCount(1);
    this->data[0].First().SetCount(1);
    core::BoundingBoxes &boxes = this->data[0].Second();
    core::misc::BezierCurvesListDataCall::Curves &curves = this->data[0].First()[0];
    vislib::RawStorage ptDataStore;
    vislib::RawStorageWriter ptData(ptDataStore);
    vislib::RawStorage idxDataStore;
    vislib::RawStorageWriter idxData(idxDataStore);
    this->hasStaticIndices = false;
    vislib::math::Cuboid<float> box;

    try {
        for (int lineIdx = 1; lineIdx < static_cast<int>(file.Count()); lineIdx++) {
            const vislib::sys::ASCIIFileBuffer::LineBuffer& line = file[lineIdx];
            if (line.Count() == 0) continue; // skip empty lines
            if (line.Word(0)[0] == '#') continue; // skip comment lines

            if (StringA(line.Word(0)).Equals("PT", false)) {
                if (line.Count() >= 8) {
                    try {
                        float x = static_cast<float>(CharTraitsA::ParseDouble(line.Word(1)));
                        float y = static_cast<float>(CharTraitsA::ParseDouble(line.Word(2)));
                        float z = static_cast<float>(CharTraitsA::ParseDouble(line.Word(3)));
                        float r = static_cast<float>(CharTraitsA::ParseDouble(line.Word(4)));
                        unsigned char colR = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(5)), 0, 255));
                        unsigned char colG = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(6)), 0, 255));
                        unsigned char colB = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(7)), 0, 255));

                        vislib::math::Cuboid<float> pbox(x - r, y - r, z - r, x + r, y + r, z + r);
                        if (ptData.Position() == 0) box = pbox;
                        else box.Union(pbox);

                        ptData << x << y << z << r << colR << colG << colB;

                    } catch(...) {
                        throw vislib::Exception("failed to parse PT", "",  lineIdx);
                    }
                    if (line.Count() > 8) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                } else throw vislib::Exception("File seems truncated", "", lineIdx);

            } else if (StringA(line.Word(0)).Equals("BC", false)) {
                if (line.Count() >= 5) {
                    try {
                        int i1 = CharTraitsA::ParseInt(line.Word(1));
                        if (i1 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                        int i2 = CharTraitsA::ParseInt(line.Word(2));
                        if (i2 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                        int i3 = CharTraitsA::ParseInt(line.Word(3));
                        if (i3 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                        int i4 = CharTraitsA::ParseInt(line.Word(4));
                        if (i4 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);

                        idxData.Write(static_cast<unsigned int>(i1));
                        idxData.Write(static_cast<unsigned int>(i2));
                        idxData.Write(static_cast<unsigned int>(i3));
                        idxData.Write(static_cast<unsigned int>(i4));

                    } catch(...) {
                        throw vislib::Exception("failed to parse BC", "",  lineIdx);
                    }
                    if (line.Count() > 5) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                } else throw vislib::Exception("File seems truncated", "", lineIdx);

            } else Log::DefaultLog.WriteWarn("Illegal command on line %d ignored", lineIdx);
        }
    } catch(...) {
        // assume frame broken
        this->data.Clear();
        throw;
    }

    unsigned int ptCnt = static_cast<unsigned int>(ptData.End()) / 19;
    if ((ptData.End() % 19) != 0) {
        Log::DefaultLog.WriteWarn("Point data seems inconsistent. Truncated to %d points", static_cast<int>(ptCnt));
    }

    unsigned int idxCnt = static_cast<unsigned int>(idxData.End() / 16) * 4u;
    if ((idxData.End() % 16) != 0) {
        Log::DefaultLog.WriteWarn("Index data seems inconsistent. Truncated to %d indices", static_cast<int>(idxCnt));
    }

    unsigned char *pd = new unsigned char[ptCnt * 19];
    unsigned int *id = new unsigned int[idxCnt];
    ::memcpy(pd, ptDataStore, ptCnt * 19);
    ::memcpy(id, idxDataStore, idxCnt * 4);

    curves.Set(core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B,
        pd, ptCnt, true, id, idxCnt, true,
        curves.GetGlobalRadius(),
        curves.GetGlobalColour()[0],
        curves.GetGlobalColour()[1],
        curves.GetGlobalColour()[2]);

    pd = nullptr; // do not delete
    id = nullptr; // do not delete

    boxes.SetObjectSpaceBBox(box);
    boxes.SetObjectSpaceClipBox(box);

}


/*
 * BezDatReader::loadASCII_2_0
 */
void BezDatReader::loadASCII_2_0(vislib::sys::ASCIIFileBuffer& file) {
    using core::misc::BezierCurvesListDataCall;
    using vislib::sys::ASCIIFileBuffer;
    using vislib::StringA;
    using vislib::sys::Log;
    using vislib::CharTraitsA;

    // parser state
    int state = 0; // pre FRAME

    try {

        // the current frame
        vislib::Pair<vislib::Array<core::misc::BezierCurvesListDataCall::Curves>,
            core::BoundingBoxes>* frame = nullptr;
        core::misc::BezierCurvesListDataCall::Curves* curves = nullptr;
        BezierCurvesListDataCall::DataLayout layout = BezierCurvesListDataCall::DATALAYOUT_NONE;
        vislib::RawStorage ptDataStore;
        vislib::RawStorageWriter ptData(ptDataStore);
        vislib::RawStorage idxDataStore;
        vislib::RawStorageWriter idxData(idxDataStore);

        for (int lineIdx = 1 /* after header */; lineIdx < static_cast<int>(file.Count()); lineIdx++) {
            const ASCIIFileBuffer::LineBuffer& line = file[lineIdx];
            if (line.Count() == 0) continue; // skip empty lines
            if (line.Word(0)[0] == '#') continue; // skip comment lines

            switch (state) {

            case 0: { // pre FRAME
                if (StringA(line.Word(0)).Equals("FRAME", false)) {
                    if (line.Count() >= 2) {
                        try {
                            int frameNum = CharTraitsA::ParseInt(line.Word(1));
                            if (frameNum != this->data.Count()) {
                                Log::DefaultLog.WriteWarn("\"FRAME %d\" expected, but found \"FRAME %d\" at line %d", static_cast<int>(this->data.Count()), frameNum, lineIdx);
                            }
                        } catch(...) {
                            throw vislib::Exception("Failed to parse frame number", "", lineIdx);
                        }
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);
                    if (line.Count() >= 3) {
                        if (StringA(line.Word(2)).Equals("STATIC_INDICES")) {
                            this->hasStaticIndices = true;
                        } else {
                            Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                        }
                        if (line.Count() > 3) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    }

                    this->data.Add(vislib::Pair<vislib::Array<core::misc::BezierCurvesListDataCall::Curves>,
                        core::BoundingBoxes>());
                    frame = &this->data[this->data.Count() - 1];

                    state = 1;
                } else Log::DefaultLog.WriteWarn("\"%s\" unexpected at line %d\n", line.Word(0), lineIdx);
            } break;

            case 1: { // in FRAME: expect BBOX, CBOX, LIST, FRAMEEND
                if (StringA(line.Word(0)).Equals("BBOX", false)) {
                    if (line.Count() >= 7) {
                        try {
                            frame->Second().SetObjectSpaceBBox(
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(1))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(2))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(3))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(4))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(5))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(6))));
                        } catch(...) {
                            throw vislib::Exception("Failed to parse BBox", "", lineIdx);
                        }
                        if (line.Count() > 7) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                } else if (StringA(line.Word(0)).Equals("CBOX", false)) {
                    if (line.Count() >= 7) {
                        try {
                            frame->Second().SetObjectSpaceClipBox(
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(1))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(2))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(3))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(4))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(5))),
                                static_cast<float>(CharTraitsA::ParseDouble(line.Word(6))));
                        } catch(...) {
                            throw vislib::Exception("Failed to parse CBox", "", lineIdx);
                        }
                        if (line.Count() > 7) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                } else if (StringA(line.Word(0)).Equals("LIST", false)) {
                    if (line.Count() >= 2) {
                        layout = BezierCurvesListDataCall::DATALAYOUT_NONE;

                        if (StringA(line.Word(1)).Equals("XYZ", false)) layout = BezierCurvesListDataCall::DATALAYOUT_XYZ_F;
                        else if (StringA(line.Word(1)).Equals("XYZR", false)) layout = BezierCurvesListDataCall::DATALAYOUT_XYZR_F;
                        else if (StringA(line.Word(1)).Equals("XYZcol", false)) layout = BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B;
                        else if (StringA(line.Word(1)).Equals("XYZRcol", false)) layout = BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B;

                        if (layout == BezierCurvesListDataCall::DATALAYOUT_NONE) {
                            Log::DefaultLog.WriteWarn("Unsupported Layout %s found at List in line %d\n", line.Word(1), lineIdx);
                            state = 2; // skip all until next LISTEND
                        } else {
                            frame->First().Add(BezierCurvesListDataCall::Curves());
                            curves = &frame->First().Last();
                            curves->SetGlobalRadius(0.5f);
                            curves->SetGlobalColour(127, 127, 127);
                            ptData.SetPosition(0);
                            ptData.SetEnd(0);
                            idxData.SetPosition(0);
                            idxData.SetEnd(0);
                            state = 3; // read line content
                        }
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                } else if (StringA(line.Word(0)).Equals("FRAMEEND", false)) {
                    // frame completed
                    frame = nullptr;
                    curves = nullptr;
                    state = 0;

                } else Log::DefaultLog.WriteWarn("\"%s\" unexpected at line %d\n", line.Word(0), lineIdx);
            } break;

            case 3: { // in LIST: search PT, BC, GLOBRAD, GLOBCOL, LISTEND
                if (StringA(line.Word(0)).Equals("PT", false)) {
                    bool hasPos = true;
                    bool hasRad = ((layout == BezierCurvesListDataCall::DATALAYOUT_XYZR_F) || (layout == BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B));
                    bool hasCol = ((layout == BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B) || (layout == BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B));

                    SIZE_T ewc = 1 + (hasPos ? 3 : 0) + (hasRad ? 1 : 0) + (hasCol ? 3 : 0);
                    if (line.Count() >= ewc) {
                        try {
                            float x = static_cast<float>(CharTraitsA::ParseDouble(line.Word(1)));
                            float y = static_cast<float>(CharTraitsA::ParseDouble(line.Word(2)));
                            float z = static_cast<float>(CharTraitsA::ParseDouble(line.Word(3)));
                            ptData.Write(x);
                            ptData.Write(y);
                            ptData.Write(z);
                            if (hasRad) {
                                float r = static_cast<float>(CharTraitsA::ParseDouble(line.Word(4)));
                                ptData.Write(r);
                            }
                            if (hasCol) {
                                unsigned char colR = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(hasRad ? 5: 4)), 0, 255));
                                unsigned char colG = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(hasRad ? 6: 5)), 0, 255));
                                unsigned char colB = static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(hasRad ? 7: 6)), 0, 255));
                                ptData.Write(colR);
                                ptData.Write(colG);
                                ptData.Write(colB);
                            }

                        } catch(...) {
                            throw vislib::Exception("Failed to parse PT", "", lineIdx);
                        }
                        if (line.Count() > ewc) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);

                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                    continue;
                } else if (StringA(line.Word(0)).Equals("BC", false)) {
                    if (line.Count() >= 5) {
                        try {
                            int i1 = CharTraitsA::ParseInt(line.Word(1));
                            if (i1 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                            int i2 = CharTraitsA::ParseInt(line.Word(2));
                            if (i2 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                            int i3 = CharTraitsA::ParseInt(line.Word(3));
                            if (i3 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);
                            int i4 = CharTraitsA::ParseInt(line.Word(4));
                            if (i4 < 0) throw vislib::Exception("negative index illegal", "", lineIdx);

                            idxData.Write(static_cast<unsigned int>(i1));
                            idxData.Write(static_cast<unsigned int>(i2));
                            idxData.Write(static_cast<unsigned int>(i3));
                            idxData.Write(static_cast<unsigned int>(i4));

                        } catch(...) {
                            throw vislib::Exception("Failed to parse PT", "", lineIdx);
                        }
                        if (line.Count() > 5) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                    continue;
                } else if (StringA(line.Word(0)).Equals("GLOBRAD", false)) {
                    if (line.Count() >= 2) {
                        try {
                            curves->SetGlobalRadius(static_cast<float>(CharTraitsA::ParseDouble(line.Word(1))));

                        } catch(...) {
                            throw vislib::Exception("Failed to parse PT", "", lineIdx);
                        }
                        if (line.Count() > 2) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                    continue;
                } else if (StringA(line.Word(0)).Equals("GLOBCOL", false)) {
                    if (line.Count() >= 4) {
                        try {
                            curves->SetGlobalColour(
                                static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(1)), 0, 255)),
                                static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(2)), 0, 255)),
                                static_cast<unsigned char>(vislib::math::Clamp(CharTraitsA::ParseInt(line.Word(3)), 0, 255)));

                        } catch(...) {
                            throw vislib::Exception("Failed to parse PT", "", lineIdx);
                        }
                        if (line.Count() > 4) Log::DefaultLog.WriteWarn("Trailing data at line %d\n", lineIdx);
                    } else throw vislib::Exception("File seems truncated", "", lineIdx);

                    continue;
                }
            } // fall through for LISTEND

            case 2: { // skip LIST: search LISTEND
                if (StringA(line.Word(0)).Equals("LISTEND", false)) {
                    unsigned int bpp;
                    switch (layout) {
                        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 12; break;
                        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 16; break;
                        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 15; break;
                        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 19; break;
                    }

                    unsigned int ptCnt = static_cast<unsigned int>(ptData.End() / bpp);
                    if ((ptData.End() % bpp) != 0) {
                        Log::DefaultLog.WriteWarn("Point data of list ending at %d seems inconsistent. Truncated to %d points", lineIdx, static_cast<int>(ptCnt));
                    }

                    unsigned int idxCnt = static_cast<unsigned int>(idxData.End() / 16) * 4u;
                    if ((idxData.End() % 16) != 0) {
                        Log::DefaultLog.WriteWarn("Index data of list ending at %d seems inconsistent. Truncated to %d indices", lineIdx, static_cast<int>(idxCnt));
                    }

                    if ((ptCnt != 0) && (idxCnt != 0)) {
                        unsigned char *pd = new unsigned char[ptCnt * bpp];
                        unsigned int *id = new unsigned int[idxCnt];
                        ::memcpy(pd, ptDataStore, ptCnt * bpp);
                        ::memcpy(id, idxDataStore, idxCnt * 4);

                        curves->Set(layout, pd, ptCnt, true, id, idxCnt, true,
                            curves->GetGlobalRadius(),
                            curves->GetGlobalColour()[0],
                            curves->GetGlobalColour()[1],
                            curves->GetGlobalColour()[2]);

                        pd = nullptr; // do not delete
                        id = nullptr; // do not delete

                    } else {
                        curves->Set(BezierCurvesListDataCall::DATALAYOUT_NONE, nullptr, 0, nullptr, 0);

                    }

                    curves = nullptr;

                    state = 1; // next list
                } else if (StringA(line.Word(0)).Equals("FRAMEEND", false)) {
                    Log::DefaultLog.WriteWarn("Unexpected FRAMEEND at line %d\n", lineIdx);
                    // frame completed
                    frame = nullptr;
                    curves = nullptr;
                    state = 0;
                }
            } break;

            default: throw vislib::Exception("Parser state exception", "", lineIdx);
            }
        }

    } catch(...) {
        // assume last frame broken
        if (this->data.Count() > 0) this->data.SetCount(this->data.Count() - 1);
        throw;
    }
}
