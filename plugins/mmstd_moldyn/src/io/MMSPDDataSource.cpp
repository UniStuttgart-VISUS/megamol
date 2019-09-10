/*
 * MMSPDDataSource.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/MMSPDDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/File.h"
#include "vislib/PtrArray.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/MemoryFile.h"
#include "vislib/memutils.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/utils.h"
#include "vislib/VersionNumber.h"

using namespace megamol;
using namespace megamol::stdplugin::moldyn::io;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.2f

/*****************************************************************************/

/*
 * MMSPDDataSource::Frame::Frame
 */
MMSPDDataSource::Frame::Frame(core::view::AnimDataModule& owner)
        : MMSPDFrameData(), core::view::AnimDataModule::Frame(owner) {
    // intentionally empty
}


/*
 * MMSPDDataSource::Frame::~Frame
 */
MMSPDDataSource::Frame::~Frame() {
    // intentionally empty
}


/*
 * MMSPDDataSource::Frame::LoadFrame
 */
bool MMSPDDataSource::Frame::LoadFrame(
        vislib::sys::File *file, unsigned int idx, UINT64 size,
        const MMSPDHeader& header, bool isBinary, bool isBigEndian) {
    this->frame = idx;
    char *buf = new char[static_cast<SIZE_T>(size)];
    try {
        if (file->Read(buf, size) != size) {
            throw vislib::Exception("Frame data truncated", __FILE__, __LINE__);
        }

        // prepare the frame object
        this->Clear();
        SIZE_T typeCnt = header.GetTypes().Count();
        this->Data().SetCount(typeCnt);
        this->IndexReconstructionData().EnforceSize(0);
        for (SIZE_T ti = 0; ti < typeCnt; ti++) {
            MMSPDFrameData::Particles& parts = this->Data()[ti];
            const MMSPDHeader::TypeDefinition& type = header.GetTypes()[ti];

            parts.SetCount(0);
            parts.Data().EnforceSize(0);
            SIZE_T fieldCnt = type.GetFields().Count();
            SIZE_T remFields = 0;
            parts.AllocateFieldMap(fieldCnt);

            // build up field map
            bool isDot = (type.GetBaseType().Equals("d", false) || type.GetBaseType().Equals("dot", false));
            bool isSphere = (type.GetBaseType().Equals("s", false) || type.GetBaseType().Equals("sphere", false));
            bool isEllipsoid = (type.GetBaseType().Equals("e", false) || type.GetBaseType().Equals("ellipsoid", false));
            bool isCylinder = (type.GetBaseType().Equals("c", false) || type.GetBaseType().Equals("cylinder", false));

            if (isDot || isSphere || isEllipsoid || isCylinder) {
                std::vector<int> idx;
                idx.resize(18);
                for (int i = 0; i < idx.size(); i++) idx[i] = -1;

                for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
                    if (type.GetFields()[fi].GetName().Equals("x"))  idx[0] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("y"))  idx[1] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("z"))  idx[2] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("cr")) idx[3] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("cg")) idx[4] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("cb")) idx[5] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("r"))  idx[6] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("rx")) idx[7] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("ry")) idx[8] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("rz")) idx[9] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("dx")) idx[10] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("dy")) idx[11] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("dz")) idx[12] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("qr")) idx[13] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("qi")) idx[14] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("qj")) idx[15] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("qk")) idx[16] = static_cast<int>(fi);
                    if (type.GetFields()[fi].GetName().Equals("ca")) idx[17] = static_cast<int>(fi);
                }

                if ((idx[0] >= 0) && (idx[1] >= 0) && (idx[2] >= 0)) {
                    // position found!
                    parts.FieldMap()[0] = idx[0];
                    parts.FieldMap()[1] = idx[1];
                    parts.FieldMap()[2] = idx[2];
                    remFields = 3;
                }

                if (isEllipsoid && (idx[7] >= 0) && (idx[8] >= 0) && (idx[9] >= 0)) {
                    // rx, ry, rz found for ellipsoid
                    parts.FieldMap()[remFields + 0] = idx[7];
                    parts.FieldMap()[remFields + 1] = idx[8];
                    parts.FieldMap()[remFields + 2] = idx[9];
                    remFields += 3;
                }

                if ((isSphere || isCylinder || (isEllipsoid && ((idx[7] < 0) || (idx[8] < 0) || (idx[9] < 0)))) && (idx[6] >= 0)) {
                    // r found
                    parts.FieldMap()[remFields] = idx[6];
                    remFields += 1;
                }

                if ((idx[10] >= 0) && (idx[11] >= 0) && (idx[12] >= 0)) {
                    // dx, dy, dz found for cylinders
                    parts.FieldMap()[remFields + 0] = idx[10];
                    parts.FieldMap()[remFields + 1] = idx[11];
                    parts.FieldMap()[remFields + 2] = idx[12];
                    remFields += 3;
                }

                if ((idx[13] >= 0) && (idx[14] >= 0) && (idx[15] >= 0) && (idx[16] >= 0)) {
                    // quaternion found
                    parts.FieldMap()[remFields + 0] = idx[13];
                    parts.FieldMap()[remFields + 1] = idx[14];
                    parts.FieldMap()[remFields + 2] = idx[15];
                    parts.FieldMap()[remFields + 3] = idx[16];
                    remFields += 4;
                }

                if ((idx[3] >= 0) && (idx[4] >= 0) && (idx[5] >= 0)) {
                    // colour found!
                    parts.FieldMap()[remFields + 0] = idx[3];
                    parts.FieldMap()[remFields + 1] = idx[4];
                    parts.FieldMap()[remFields + 2] = idx[5];
                    remFields += 3;
                    if (idx[17] >= 0) {
                        parts.FieldMap()[remFields + 0] = idx[17];
                        remFields++;
                    }
                }

            }

            if (remFields < fieldCnt) {
                for (SIZE_T fi = 0; fi < fieldCnt; fi++) { // 1:1 map
                    bool found = false;
                    for (SIZE_T i = 0; i < remFields; i++) {
                        if (parts.FieldMap()[i] == fi) {
                            found = true;
                            break;
                        }
                    }
                    if (found) continue; // field already stored
                    parts.FieldMap()[remFields] = static_cast<unsigned int>(fi);
                    remFields++;
                }
            }

        }

        // now actually load the data
        if (isBinary) {
            if (!isBigEndian) {
                this->loadFrameBinary(buf, size, header);
            } else {
                this->loadFrameBinaryBE(buf, size, header);
            }
        } else {
            this->loadFrameText(buf, size, header);
        }

        for (SIZE_T ti = 0; ti < typeCnt; ti++) {
            MMSPDFrameData::Particles& parts = this->Data()[ti];
            SIZE_T ps = header.GetTypes()[ti].GetFields().Count() * sizeof(float);
            // TODO: decide whether to ignore IDs or don't! (current: don't, interleave with pos)
            if (header.HasIDs()) ps += 8;
            ASSERT((parts.GetData().GetSize() % ps) == 0);
            parts.SetCount(parts.GetData().GetSize() / ps);
            if ((parts.Count() > 1) && (header.HasIDs())) {
                // we sort particles here!
                ::qsort(parts.Data(), parts.GetData().GetSize() / ps, ps,
                    [](const void* a, const void* b) -> int {
                        return static_cast<int>(static_cast<const uint64_t*>(a)[0] - static_cast<const uint64_t*>(b)[0]);
                    });
            }
        }

    } catch (...) {
        delete[] buf;
        this->Clear();
        throw;
        //return false;
    }

    delete[] buf;
    return true;
}


/*
 * MMSPDDataSource::Frame::SetData
 */
void MMSPDDataSource::Frame::SetData(core::moldyn::MultiParticleDataCall& call,
        const MMSPDHeader& header) {
    call.SetParticleListCount(static_cast<unsigned int>(this->Data().Count()));
    for (SIZE_T pi = 0; pi < this->Data().Count(); pi++) {
        core::moldyn::MultiParticleDataCall::Particles &pts = call.AccessParticles(static_cast<unsigned int>(pi));
        const MMSPDHeader::TypeDefinition &typeDef = header.GetTypes()[pi];
        MMSPDFrameData::Particles &parts = this->Data()[pi];

        bool hasX = false, hasY = false, hasZ = false, hasR = false, hasCR = false, hasCG = false, hasCB = false, hasCA = false;
        unsigned int rIdx = -1, cIdx = -1, pIdx = -1;
        for (SIZE_T fi = 0; fi < typeDef.GetFields().Count(); fi++) {
            const vislib::StringA& fn = typeDef.GetFields()[fi].GetName();
            if (fn.Equals("x")) { hasX = true; pIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("y")) hasY = true;
            if (fn.Equals("z")) hasZ = true;
            if (fn.Equals("r")) { hasR = true; rIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("cr")) { hasCR = true; cIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("cg")) hasCG = true;
            if (fn.Equals("cb")) hasCB = true;
            if (fn.Equals("ca")) hasCA = true;
        }

        if (hasR && (parts.FieldMap()[rIdx] != 3)) {
            // radius vector stored. ... crap
            hasR = false;
        }

        pts.SetGlobalColour(191, 191, 191);
        pts.SetGlobalRadius(0.5f);
        unsigned int cr = 191, cg = 191, cb = 191, ca = 255;
        for (SIZE_T fi = 0; fi < typeDef.GetConstFields().Count(); fi++) {
            const MMSPDHeader::ConstField &f = typeDef.GetConstFields()[fi];
            if (f.GetName().Equals("r")) pts.SetGlobalRadius(f.GetAsFloat());
            if (f.GetName().Equals("cr")) pts.SetGlobalColour(cr = static_cast<unsigned int>(f.GetAsFloat() * 255.0f), cg, cb);
            if (f.GetName().Equals("cg")) pts.SetGlobalColour(cr, cg = static_cast<unsigned int>(f.GetAsFloat() * 255.0f), cb);
            if (f.GetName().Equals("cb")) pts.SetGlobalColour(cr, cg, cb = static_cast<unsigned int>(f.GetAsFloat() * 255.0f));
            if (f.GetName().Equals("ca")) pts.SetGlobalColour(cr, cg, cb, ca = static_cast<unsigned int>(f.GetAsFloat() * 255.0f));
        }

        // now use some because-I-know-magic:
        unsigned int off = 0;
        if (!hasX || !hasY || !hasZ || parts.Count() == 0) {
            // too empty
            pts.SetCount(0);
            continue;

        } else {
            pts.SetCount(parts.Count());

            // TODO: decide whether to ignore IDs or don't! (current: don't, interleave with pos)
            if (header.HasIDs()) {
                off += 8;
                pts.SetIDData(core::moldyn::MultiParticleDataCall::Particles::IDDATA_UINT64,
                    parts.Data().At(0),
                    static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
            }

            pts.SetVertexData(hasR ? core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR
                : core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                parts.Data().At(off + pIdx * sizeof(float)),
                static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
            //printf("%lu\n", parts.Count());
            //printf("%f %f %f\n", parts.Data().AsAt<float>(off)[0], parts.Data().AsAt<float>(off)[1], parts.Data().AsAt<float>(off)[2]);

            if (hasCR && hasCG && hasCB) {
                if (hasCA) {
                    pts.SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA,
                        parts.Data().At(off + cIdx * sizeof(float)),
                        static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
                } else {
                    pts.SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB,
                        parts.Data().At(off + cIdx * sizeof(float)),
                        static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
                }
            } else {
                pts.SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            }

        }

    }

}


/*
 * MMSPDDataSource::Frame::SetDirData
 */
void MMSPDDataSource::Frame::SetDirData(core::moldyn::MultiParticleDataCall& call,
    const MMSPDHeader& header) {
    call.SetParticleListCount(static_cast<unsigned int>(this->Data().Count()));
    for (SIZE_T pi = 0; pi < this->Data().Count(); pi++) {
        auto &pts = call.AccessParticles(static_cast<unsigned int>(pi));
        const MMSPDHeader::TypeDefinition &typeDef = header.GetTypes()[pi];
        MMSPDFrameData::Particles &parts = this->Data()[pi];

        bool hasX = false, hasY = false, hasZ = false, hasR = false, hasCR = false, hasCG = false, hasCB = false, hasCA = false, hasDX = false, hasDY = false, hasDZ = false;
        unsigned int rIdx = -1, cIdx = -1, dIdx = -1, pIdx = -1;
        for (SIZE_T fi = 0; fi < typeDef.GetFields().Count(); fi++) {
            const vislib::StringA& fn = typeDef.GetFields()[fi].GetName();
            if (fn.Equals("x")) { hasX = true; pIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("y")) hasY = true;
            if (fn.Equals("z")) hasZ = true;
            if (fn.Equals("r")) { hasR = true; rIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("cr")) { hasCR = true; cIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("cg")) hasCG = true;
            if (fn.Equals("cb")) hasCB = true;
            if (fn.Equals("ca")) hasCA = true;
            if (fn.Equals("dx")) { hasDX = true; dIdx = static_cast<unsigned int>(fi); }
            if (fn.Equals("dy")) hasDY = true;
            if (fn.Equals("dz")) hasDZ = true;
        }

        if (hasR && (parts.FieldMap()[rIdx] != 3)) {
            // radius vector stored. ... crap
            hasR = false;
        }

        pts.SetGlobalColour(191, 191, 191);
        pts.SetGlobalRadius(0.5f);
        unsigned int cr = 191, cg = 191, cb = 191, ca = 255;
        for (SIZE_T fi = 0; fi < typeDef.GetConstFields().Count(); fi++) {
            const MMSPDHeader::ConstField &f = typeDef.GetConstFields()[fi];
            if (f.GetName().Equals("r")) pts.SetGlobalRadius(f.GetAsFloat());
            if (f.GetName().Equals("cr")) pts.SetGlobalColour(cr = static_cast<unsigned int>(f.GetAsFloat() * 255.0f), cg, cb);
            if (f.GetName().Equals("cg")) pts.SetGlobalColour(cr, cg = static_cast<unsigned int>(f.GetAsFloat() * 255.0f), cb);
            if (f.GetName().Equals("cb")) pts.SetGlobalColour(cr, cg, cb = static_cast<unsigned int>(f.GetAsFloat() * 255.0f));
            if (f.GetName().Equals("ca")) pts.SetGlobalColour(cr, cg, cb, ca = static_cast<unsigned int>(f.GetAsFloat() * 255.0f));
        }

        // now use some because-I-know-magic:
        unsigned int off = 0;
        
        if (!hasX || !hasY || !hasZ || parts.Count() == 0) {
            // too empty
            pts.SetCount(0);
            continue;

        } else {
            pts.SetCount(parts.Count());

            // TODO: decide whether to ignore IDs or don't! (current: don't, interleave with pos)
            if (header.HasIDs()) {
                off += 8;
                pts.SetIDData(core::moldyn::SimpleSphericalParticles::IDDATA_UINT64,
                    parts.Data().At(0),
                    static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
            }

            pts.SetVertexData(hasR ? core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR
                                   : core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ,
                parts.Data().At(off + pIdx * sizeof(float)),
                static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));

            if (hasDX && hasDY && hasDZ) {
                pts.SetDirData(core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ,
                    parts.Data().At(off + dIdx * sizeof(float)),
                    static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
            } else {
                pts.SetDirData(core::moldyn::SimpleSphericalParticles::DIRDATA_NONE,
                    nullptr);
            }

            //printf("%lu\n", parts.Count());
            //printf("%f %f %f\n", parts.Data().AsAt<float>(off)[0], parts.Data().AsAt<float>(off)[1], parts.Data().AsAt<float>(off)[2]);

            if (hasCR && hasCG && hasCB) {
                if (hasCA) {
                    pts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA,
                        parts.Data().At(off + cIdx * sizeof(float)),
                        static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
                } else {
                    pts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB,
                        parts.Data().At(off + cIdx * sizeof(float)),
                        static_cast<unsigned int>(off + typeDef.GetFields().Count() * sizeof(float)));
                }
            } else {
                pts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_NONE, NULL);
            }

        }

    }

}


/*
 * MMSPDDataSource::Frame::loadFrameText
 */
void MMSPDDataSource::Frame::loadFrameText(char *buffer, UINT64 size, const MMSPDHeader& header) {
    // We don't have to brother with unicode here, because there is no string data allowed.
    // All characters must be white space, line breaks, '>' and characters forming numbers (digits, dots, plus, minus, 'e').
    vislib::sys::MemoryFile mem;
    mem.Open(static_cast<void*>(buffer), static_cast<SIZE_T>(size), vislib::sys::File::READ_ONLY);
    vislib::sys::ASCIIFileBuffer txt(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!txt.LoadFile(mem)) throw vislib::Exception(__FILE__, __LINE__);

    if (txt.Count() <= 0) throw vislib::Exception("Unable to load frame data", __FILE__, __LINE__);
    if (txt.Line(0).Count() < 2) throw vislib::Exception("Illegal time frame marker", __FILE__, __LINE__);
    UINT64 partCnt = vislib::CharTraitsA::ParseUInt64(txt.Line(0).Word(1));
    if (txt.Count() < partCnt + 1) throw vislib::Exception("Data frame truncated", __FILE__, __LINE__);

    SIZE_T typeCnt = header.GetTypes().Count();
    vislib::PtrArray<vislib::RawStorageWriter> typeData;
    typeData.SetCount(typeCnt);
    for (SIZE_T i = 0; i < typeCnt; i++) {
        typeData[i] = new vislib::RawStorageWriter(this->Data()[i].Data());
        typeData[i]->SetIncrement(vislib::math::Max<SIZE_T>(
            static_cast<SIZE_T>((header.GetTypes()[i].GetFields().Count() * sizeof(float) * partCnt)
            / (2 * typeCnt)),
            1024 * 1024));
    }
    vislib::RawStorageWriter idxRecDat(this->IndexReconstructionData());
    if (typeCnt > 1) idxRecDat.SetIncrement(vislib::math::Max<SIZE_T>(static_cast<SIZE_T>(partCnt / 10), 10 * 1024));
    UINT32 irdLastType = static_cast<UINT32>(typeCnt);
    UINT64 irdLastCount;

    SIZE_T type = 0;
    for (UINT64 pi = 0; pi < partCnt; pi++) {
        const vislib::sys::ASCIIFileBuffer::LineBuffer &line = txt.Line(static_cast<SIZE_T>(1 + pi));
        unsigned int off = 0;
        if (typeCnt > 1) {
            if (header.HasIDs()) {
                if (line.Count() < 2) throw vislib::Exception("line truncated", __FILE__, __LINE__);
                type = static_cast<SIZE_T>(vislib::CharTraitsA::ParseInt(line.Word(1)));
                if (type >= typeCnt) throw vislib::Exception("Illegal type encountered", __FILE__, __LINE__);
                typeData[type]->Write(vislib::CharTraitsA::ParseUInt64(line.Word(0)));
                off = 2;
            } else {
                if (line.Count() < 1) throw vislib::Exception("line truncated", __FILE__, __LINE__);
                type = static_cast<SIZE_T>(vislib::CharTraitsA::ParseInt(line.Word(0)));
                if (type >= typeCnt) throw vislib::Exception("Illegal type encountered", __FILE__, __LINE__);
                off = 1;
            }
        } else if (header.HasIDs()) {
            // type remains 0
            if (line.Count() < 1) throw vislib::Exception("line truncated", __FILE__, __LINE__);
            typeData[type]->Write(vislib::CharTraitsA::ParseUInt64(line.Word(0)));
            off = 1;
        }

        this->addIndexForReconstruction(static_cast<UINT32>(type), idxRecDat,
            this->IndexReconstructionData(), irdLastType, irdLastCount);

        SIZE_T fieldCnt = header.GetTypes()[type].GetFields().Count();
        if (line.Count() < fieldCnt + off) throw vislib::Exception("line truncated", __FILE__, __LINE__);

        for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
            //float val = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(off + this->GetData()[type].FieldMap()[fi])));
            //if (header.GetTypes()[type].GetFields()[this->GetData()[type].FieldMap()[fi]].GetType() ==
            //    MMSPDHeader::Field::TYPE_BYTE) {
            //    val /= 255.0f;
            //}
            float val = vislib::CharTraitsA::ParseDouble(line.Word(off + fi));
            if (header.GetTypes()[type].GetFields()[fi].GetType() == MMSPDHeader::Field::TYPE_BYTE) {
                val /= 255.0f;
            }

            //vislib::sys::Log::DefaultLog.WriteInfo("read: %f", val);
            typeData[type]->Write(val);
        }

    }

    for (SIZE_T i = 0; i < typeCnt; i++) {
        this->Data()[i].Data().EnforceSize(typeData[i]->End(), true);
    }
    this->IndexReconstructionData().EnforceSize(idxRecDat.End(), true);
}


/*
 * MMSPDDataSource::Frame::loadFrameBinary
 */
void MMSPDDataSource::Frame::loadFrameBinary(char *buffer, UINT64 size, const MMSPDHeader& header) {
    UINT64 &partCnt = *reinterpret_cast<UINT64*>(&buffer[0]);
    SIZE_T pos = 8;
    SIZE_T typeCnt = header.GetTypes().Count();
    vislib::PtrArray<vislib::RawStorageWriter> typeData;
    typeData.SetCount(typeCnt);
    SIZE_T valuesCount = 0;
    for (SIZE_T i = 0; i < typeCnt; i++) {
        typeData[i] = new vislib::RawStorageWriter(this->Data()[i].Data());
        valuesCount = vislib::math::Max(valuesCount, header.GetTypes()[i].GetFields().Count());
        typeData[i]->SetIncrement(vislib::math::Max<SIZE_T>(
            static_cast<SIZE_T>((header.GetTypes()[i].GetFields().Count() * sizeof(float) * partCnt)
            / (2 * typeCnt)),
            1024 * 1024));
    }
    vislib::SmartPtr<float, vislib::ArrayAllocator<float> > values = new float[valuesCount];
    vislib::RawStorageWriter idxRecDat(this->IndexReconstructionData());
    if (typeCnt > 1) idxRecDat.SetIncrement(vislib::math::Max<SIZE_T>(static_cast<SIZE_T>(partCnt / 10), 10 * 1024));
    UINT32 irdLastType = static_cast<UINT32>(typeCnt);
    UINT64 irdLastCount;

    SIZE_T type = 0;
    for (UINT64 pi = 0; pi < partCnt; pi++) {

        // TODO: decide whether to ignore IDs or don't! (current: don't, interleave with pos) [#128]
        if (header.HasIDs()) pos += 8;

        if (typeCnt > 1) {
            type = *reinterpret_cast<UINT32*>(&buffer[pos]);
            if (type >= typeCnt) throw vislib::Exception("Illegal type encountered", __FILE__, __LINE__);
            if (header.HasIDs()) {
                typeData[type]->Write(*reinterpret_cast<UINT64*>(&buffer[pos - 8]));
            }
            pos += 4;
        } else {
            if (header.HasIDs()) {
                typeData[type]->Write(*reinterpret_cast<UINT64*>(&buffer[pos - 8]));
            }
        }

        this->addIndexForReconstruction(static_cast<UINT32>(type), idxRecDat,
            this->IndexReconstructionData(), irdLastType, irdLastCount);

        SIZE_T fieldCnt = header.GetTypes()[type].GetFields().Count();

        for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
            MMSPDHeader::Field::TypeID typeID = header.GetTypes()[type].GetFields()[fi].GetType();
            switch (typeID) {
            case MMSPDHeader::Field::TYPE_BYTE:
                values.operator->()[fi] = static_cast<float>(
                    *reinterpret_cast<unsigned char*>(&buffer[pos])) / 255.0f;
                pos += 1;
                break;
            case MMSPDHeader::Field::TYPE_FLOAT:
                values.operator->()[fi] = *reinterpret_cast<float*>(&buffer[pos]);
                pos += 4;
                break;
            case MMSPDHeader::Field::TYPE_DOUBLE:
                values.operator->()[fi] = static_cast<float>(
                    *reinterpret_cast<double*>(&buffer[pos]));
                pos += 8;
                break;
            }
        }

        for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
            //typeData[type]->Write(values.operator->()[this->GetData()[type].FieldMap()[fi]]);
            typeData[type]->Write(values.operator->()[fi]);
        }

    }

    for (SIZE_T i = 0; i < typeCnt; i++) {
        this->Data()[i].Data().EnforceSize(typeData[i]->End(), true);
    }
    this->IndexReconstructionData().EnforceSize(idxRecDat.End(), true);

}


/*
 * MMSPDDataSource::Frame::loadFrameBinaryBE
 */
void MMSPDDataSource::Frame::loadFrameBinaryBE(char *buffer, UINT64 size, const MMSPDHeader& header) {
    UINT64 &partCnt = *reinterpret_cast<UINT64*>(&buffer[0]);
    SIZE_T pos = 8;
    SIZE_T typeCnt = header.GetTypes().Count();
    vislib::PtrArray<vislib::RawStorageWriter> typeData;
    typeData.SetCount(typeCnt);
    SIZE_T valuesCount = 0;
    for (SIZE_T i = 0; i < typeCnt; i++) {
        typeData[i] = new vislib::RawStorageWriter(this->Data()[i].Data());
        valuesCount = vislib::math::Max(valuesCount, header.GetTypes()[i].GetFields().Count());
        typeData[i]->SetIncrement(vislib::math::Max<SIZE_T>(
            static_cast<SIZE_T>((header.GetTypes()[i].GetFields().Count() * sizeof(float) * partCnt)
            / (2 * typeCnt)),
            1024 * 1024));
    }
    vislib::SmartPtr<float, vislib::ArrayAllocator<float> > values = new float[valuesCount];
    vislib::RawStorageWriter idxRecDat(this->IndexReconstructionData());
    if (typeCnt > 1) idxRecDat.SetIncrement(vislib::math::Max<SIZE_T>(static_cast<SIZE_T>(partCnt / 10), 10 * 1024));
    UINT32 irdLastType = static_cast<UINT32>(typeCnt);
    UINT64 irdLastCount;

    SIZE_T type = 0;
    for (UINT64 pi = 0; pi < partCnt; pi++) {

        // see loadframe
        if (header.HasIDs()) pos += 8;

        if (typeCnt > 1) {
            vislib::Swap(buffer[pos + 0], buffer[pos + 3]);
            vislib::Swap(buffer[pos + 1], buffer[pos + 2]);
            type = *reinterpret_cast<UINT32*>(&buffer[pos]);
            if (type >= typeCnt) throw vislib::Exception("Illegal type encountered", __FILE__, __LINE__);
            // see loadframe
            if (header.HasIDs()) {
                vislib::Swap(buffer[pos - 8 + 0], buffer[pos - 8 + 7]);
                vislib::Swap(buffer[pos - 8 + 1], buffer[pos - 8 + 6]);
                vislib::Swap(buffer[pos - 8 + 2], buffer[pos - 8 + 5]);
                vislib::Swap(buffer[pos - 8 + 3], buffer[pos - 8 + 4]);
                typeData[type]->Write(*reinterpret_cast<UINT64*>(&buffer[pos - 8]));
            }
            pos += 4;
        } else {
            if (header.HasIDs()) {
                vislib::Swap(buffer[pos - 8 + 0], buffer[pos - 8 + 7]);
                vislib::Swap(buffer[pos - 8 + 1], buffer[pos - 8 + 6]);
                vislib::Swap(buffer[pos - 8 + 2], buffer[pos - 8 + 5]);
                vislib::Swap(buffer[pos - 8 + 3], buffer[pos - 8 + 4]);
                typeData[type]->Write(*reinterpret_cast<UINT64*>(&buffer[pos - 8]));
            }
        }

        this->addIndexForReconstruction(static_cast<UINT32>(type), idxRecDat,
            this->IndexReconstructionData(), irdLastType, irdLastCount);

        SIZE_T fieldCnt = header.GetTypes()[type].GetFields().Count();

        for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
            MMSPDHeader::Field::TypeID typeID = header.GetTypes()[type].GetFields()[fi].GetType();
            switch (typeID) {
            case MMSPDHeader::Field::TYPE_BYTE:
                values.operator->()[fi] = static_cast<float>(
                    *reinterpret_cast<unsigned char*>(&buffer[pos])) / 255.0f;
                pos += 1;
                break;
            case MMSPDHeader::Field::TYPE_FLOAT:
                vislib::Swap(buffer[pos + 0], buffer[pos + 3]);
                vislib::Swap(buffer[pos + 1], buffer[pos + 2]);
                values.operator->()[fi] = *reinterpret_cast<float*>(&buffer[pos]);
                pos += 4;
                break;
            case MMSPDHeader::Field::TYPE_DOUBLE:
                vislib::Swap(buffer[pos + 0], buffer[pos + 7]);
                vislib::Swap(buffer[pos + 1], buffer[pos + 6]);
                vislib::Swap(buffer[pos + 2], buffer[pos + 5]);
                vislib::Swap(buffer[pos + 3], buffer[pos + 4]);
                values.operator->()[fi] = static_cast<float>(
                    *reinterpret_cast<double*>(&buffer[pos]));
                pos += 8;
                break;
            }
        }

        for (SIZE_T fi = 0; fi < fieldCnt; fi++) {
            //typeData[type]->Write(values.operator->()[this->GetData()[type].FieldMap()[fi]]);
            typeData[type]->Write(values.operator->()[fi]);
        }

    }

    for (SIZE_T i = 0; i < typeCnt; i++) {
        this->Data()[i].Data().EnforceSize(typeData[i]->End(), true);
    }
    this->IndexReconstructionData().EnforceSize(idxRecDat.End(), true);

}


/*
 * MMSPDDataSource::Frame::addIndexForReconstruction
 */
void MMSPDDataSource::Frame::addIndexForReconstruction(UINT32 type,
        class vislib::RawStorageWriter& wrtr, class vislib::RawStorage& data,
        UINT32 &lastType, UINT64 &lastCount) {
    unsigned char dat[10];
    unsigned int datLen;

    if (type != lastType) {
        lastType = type;
        lastCount = 1;
        datLen = 10;
        if (!vislib::UIntRLEEncode(dat, datLen, type)) throw vislib::Exception(__FILE__, __LINE__);
        wrtr.Write(dat, datLen);
        datLen = 10;
        if (!vislib::UIntRLEEncode(dat, datLen, lastCount)) throw vislib::Exception(__FILE__, __LINE__);
        wrtr.Write(dat, datLen);

    } else {
        wrtr.SetPosition(wrtr.Position() - vislib::UIntRLELength(lastCount));
        datLen = 10;
        lastCount++;
        if (!vislib::UIntRLEEncode(dat, datLen, lastCount)) throw vislib::Exception(__FILE__, __LINE__);
        wrtr.Write(dat, datLen);

    }

}

/*****************************************************************************/

/*
 * MMSPDDataSource::FileFormatAutoDetect
 */
float MMSPDDataSource::FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize) {
    return (((dataSize >= 6)
        && ((::memcmp(data, "MMSPDb", 6) == 0)
            || (::memcmp(data, "MMSPDa", 6) == 0)
            || (::memcmp(data, "MMSPDu", 6) == 0)))
        || ((dataSize >= 9)
        && (::memcmp(data, "\xEF\xBB\xBFMMSPDu", 9) == 0))) ? 1.0f : 0.0f;
}


/*
 * MMSPDDataSource::MMSPDDataSource
 */
MMSPDDataSource::MMSPDDataSource(void)
    : core::view::AnimDataModule()
    , filename("filename", "The path to the MMSPD file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , getDirData("getdirdata", "(optional) Slot to request directional data from this data source.")
    , dataHeader(), file(NULL), frameIdx(NULL)
    , clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), isBinaryFile(true)
    , isBigEndian(false), frameIdxLock(), frameIdxEvent(true)
    , frameIdxThread(&MMSPDDataSource::buildFrameIndex)
    , dataHash(0) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMSPDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &MMSPDDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &MMSPDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->getDirData.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0),
        &MMSPDDataSource::getDirDataCallback);
    this->getDirData.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1),
        &MMSPDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDirData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * MMSPDDataSource::~MMSPDDataSource
 */
MMSPDDataSource::~MMSPDDataSource(void) {
    this->Release();
}


/*
 * MMSPDDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* MMSPDDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<MMSPDDataSource*>(this));
    return f;
}


/*
 * MMSPDDataSource::create
 */
bool MMSPDDataSource::create(void) {
    return true;
}


/*
 * MMSPDDataSource::loadFrame
 */
void MMSPDDataSource::loadFrame(core::view::AnimDataModule::Frame *frame,
        unsigned int idx) {
    using vislib::sys::Log;
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        //f->Clear();
        return;
    }
    //printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    ASSERT(idx < this->FrameCount());

    // ask frame index for seek positions
    UINT64 firstSeek = 1, fromSeek = 0, toSeek = 0;

    while ((fromSeek == 0) || (toSeek == 0)) {

        this->frameIdxLock.Lock();
        if (this->frameIdx == NULL) {
            firstSeek = 0;
        } else {
            firstSeek = this->frameIdx[0];
            fromSeek = this->frameIdx[idx];
            toSeek = this->frameIdx[idx + 1];
        }
        if ((firstSeek != 0) && ((fromSeek == 0) || (toSeek == 0))) {
            this->frameIdxEvent.Reset();
        }
        this->frameIdxLock.Unlock();

        if (firstSeek == 0) {
            // frame index generation failed
            f->Clear();
            return;
        }
        if ((fromSeek == 0) || (toSeek == 0)) {
            this->frameIdxEvent.Wait();
        }

    }

    this->file->Seek(fromSeek);
    bool res = false;
    vislib::StringA errMsg;
    try {
        res = f->LoadFrame(this->file, idx, toSeek - fromSeek,
            this->dataHeader, this->isBinaryFile, this->isBigEndian);
        if (!res) errMsg = "Unknown error";
    } catch(vislib::Exception e) {
        errMsg.Format("%s [%s:%d]", e.GetMsgA(), e.GetFile(), e.GetLine());
        res = false;
    } catch(...) {
        errMsg = "Unknown exception";
        res = false;
    }
    if (!res) {
        // failed
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMSPD file: %s",
            idx, errMsg.PeekBuffer());
    }
}


/*
 * MMSPDDataSource::release
 */
void MMSPDDataSource::release(void) {
    this->clearData();
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File *f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    ARY_SAFE_DELETE(this->frameIdx);
}


/*
 *MMSPDDataSource::buildFrameIndex
 */
DWORD MMSPDDataSource::buildFrameIndex(void *userdata) {
    MMSPDDataSource *that = static_cast<MMSPDDataSource *>(userdata);
    vislib::sys::File f;
    if (!f.Open(that->filename.Param<core::param::FilePathParam>()->Value(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        that->frameIdxLock.Lock();
        that->frameIdx[0] = 0;
        vislib::sys::Log::DefaultLog.WriteError("Unable to open data file a second time for frame index generation");
        that->frameIdxLock.Unlock();
        that->frameIdxEvent.Set();
        return 0;
    }
    f.Seek(that->frameIdx[0]); // lock not required, because i know the main thread is currently waiting to load the first frame
    vislib::sys::Log::DefaultLog.WriteInfo(50, "Frame index generation started.");

    const SIZE_T MAX_BUFFER_SIZE = 1024 * 1024;
    char *buffer = new char[MAX_BUFFER_SIZE];
    unsigned int frameCount = that->dataHeader.GetTimeCount();
    unsigned int frame = 0;
    vislib::StringA token;

    // sizes of a particle in binary files
    unsigned int *typeSizes = new unsigned int[that->dataHeader.GetTypes().Count()];
    for (int i = 0; i < static_cast<int>(that->dataHeader.GetTypes().Count()); i++) {
        typeSizes[i] = that->dataHeader.GetTypes()[i].GetDataSize();
        if (that->dataHeader.HasIDs()) typeSizes[i] += 8;
        if (that->dataHeader.GetTypes().Count() > 1) typeSizes[i] += 4;
    }

    try {

        if (that->isBinaryFile && (that->dataHeader.GetTypes().Count() == 1)) {
            // binary, but only one type! This is perfect
            UINT64 partCnt;

            for (frame = 0; frame < frameCount; frame++) {
                that->frameIdxLock.Lock();
                if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
                that->frameIdx[frame] = static_cast<UINT64>(f.Tell());
                that->frameIdxEvent.Set();
                that->frameIdxLock.Unlock();
                if (f.Read(&partCnt, 8) != 8) {
                    // file truncated, so mark these frames as to be empty
                    that->frameIdxLock.Lock();
                    for (; frame <= frameCount; frame++) {
                        that->frameIdx[frame] = ULLONG_MAX;
                    }
                    that->frameIdxEvent.Set();
                    that->frameIdxLock.Unlock();
                    break;

                }
                if (that->isBigEndian) {
                    unsigned char *fac = reinterpret_cast<unsigned char*>(&partCnt);
                    vislib::Swap(fac[0], fac[7]);
                    vislib::Swap(fac[1], fac[6]);
                    vislib::Swap(fac[2], fac[5]);
                    vislib::Swap(fac[3], fac[4]);
                }

                if ((partCnt != that->dataHeader.GetParticleCount()) && (that->dataHeader.GetParticleCount() != 0)) {
                    throw new vislib::Exception("Particle count changed between frames even the header already defined the count", __FILE__, __LINE__);
                }

                // now skip the actual data
                f.Seek(partCnt * typeSizes[0], vislib::sys::File::CURRENT);

            }

            that->frameIdxLock.Lock();
            if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
            that->frameIdx[frame] = static_cast<UINT64>(f.Tell());
            that->frameIdxEvent.Set();
            that->frameIdxLock.Unlock();

        } else {
            unsigned int parserState = 0;
            UINT64 framePartCnt;
            UINT64 partIdx = 0;

            while (!f.IsEOF()) {
                UINT64 bufPos = f.Tell();
                SIZE_T bufSize = static_cast<SIZE_T>(f.Read(buffer, MAX_BUFFER_SIZE));
                SIZE_T bufIdx = 0;

                if (that->isBinaryFile) {
                    // binary, but with several types
                    bool loadNext = false;

                    while (!loadNext) {
                        switch (parserState) {
                        case 0: { // reading particle count
                            if ((bufSize - bufIdx) < 8) { // insufficient data in buffer
                                if (f.IsEOF()) {
                                    that->frameIdxLock.Lock();
                                    for (; frame <= frameCount; frame++) {
                                        that->frameIdx[frame] = ULLONG_MAX;
                                    }
                                    that->frameIdxEvent.Set();
                                    that->frameIdxLock.Unlock();
                                    break;
                                }
                                f.Seek(bufIdx - bufSize, vislib::sys::File::CURRENT); // step a bit back
                                loadNext = true;
                                continue; // reload the buffer
                            }

                            that->frameIdxLock.Lock();
                            if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
                            that->frameIdx[frame++] = bufPos + bufIdx;
                            that->frameIdxEvent.Set();
                            that->frameIdxLock.Unlock();

                            framePartCnt = *reinterpret_cast<UINT64*>(&buffer[bufIdx]);
                            if (that->isBigEndian) {
                                unsigned char *fac = reinterpret_cast<unsigned char*>(&framePartCnt);
                                vislib::Swap(fac[0], fac[7]);
                                vislib::Swap(fac[1], fac[6]);
                                vislib::Swap(fac[2], fac[5]);
                                vislib::Swap(fac[3], fac[4]);
                            }
                            bufIdx += 8;
                            if ((framePartCnt != that->dataHeader.GetParticleCount()) && (that->dataHeader.GetParticleCount() != 0)) {
                                throw new vislib::Exception("Particle count changed between frames even the header already defined the count", __FILE__, __LINE__);
                            }
                            partIdx = 0;
                            if (framePartCnt > 0) {
                                parserState = that->dataHeader.HasIDs() ? 1 : 2;
                            }

                        } break;

                        case 1: { // reading a particle ID
                            if ((bufSize - bufIdx) < 8) { // insufficient data in buffer
                                if (f.IsEOF()) {
                                    that->frameIdxLock.Lock();
                                    for (; frame <= frameCount; frame++) {
                                        that->frameIdx[frame] = ULLONG_MAX;
                                    }
                                    that->frameIdxEvent.Set();
                                    that->frameIdxLock.Unlock();
                                    break;
                                }
                                f.Seek(bufIdx - bufSize, vislib::sys::File::CURRENT); // step a bit back
                                loadNext = true;
                                continue; // reload the buffer
                            }
                            bufIdx += 8;
                            parserState = 2;

                        } break;

                        case 2: { // reading a particle type
                            if ((bufSize - bufIdx) < 4) { // insufficient data in buffer
                                if (f.IsEOF()) {
                                    that->frameIdxLock.Lock();
                                    for (; frame <= frameCount; frame++) {
                                        that->frameIdx[frame] = ULLONG_MAX;
                                    }
                                    that->frameIdxEvent.Set();
                                    that->frameIdxLock.Unlock();
                                    break;
                                }
                                f.Seek(bufIdx - bufSize, vislib::sys::File::CURRENT); // step a bit back
                                loadNext = true;
                                continue; // reload the buffer
                            }
                            unsigned int type = *reinterpret_cast<UINT32*>(&buffer[bufIdx]);
                            if (that->isBigEndian) {
                                unsigned char *fac = reinterpret_cast<unsigned char*>(&type);
                                vislib::Swap(fac[0], fac[3]);
                                vislib::Swap(fac[1], fac[2]);
                            }
                            if (type >= that->dataHeader.GetTypes().Count()) {
                                vislib::StringA msg;
                                msg.Format("Illegal type value %u/%u read", type, static_cast<unsigned int>(that->dataHeader.GetTypes().Count()));
                                throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
                            }
                            bufIdx += 4;

                            unsigned int size = typeSizes[type] - 4; // -4 for the type
                            if (that->dataHeader.HasIDs()) size -= 8;

                            // now skip 'size' bytes
                            if ((bufIdx + size) < bufSize) {
                                bufIdx += size;
                            } else {
                                size -= static_cast<unsigned int>(bufSize - bufIdx);
                                f.Seek(size, vislib::sys::File::CURRENT);
                                bufPos = f.Tell();
                                bufIdx = 0;
                                loadNext = true;
                            }
                            partIdx++;
                            if (partIdx == framePartCnt) {
                                if (frame == frameCount) {
                                    that->frameIdxLock.Lock();
                                    if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
                                    that->frameIdx[frame++] = bufPos + bufIdx;
                                    that->frameIdxEvent.Set();
                                    that->frameIdxLock.Unlock();
                                    loadNext = true;
                                }
                                parserState = 0;
                            } else if (that->dataHeader.HasIDs()) {
                                parserState = 1;
                            }

                        } break;

                        }
                    }

                    if (frame > frameCount) {
                        break;
                    }

                } else {
                    // text, everything is lost
                    for (; bufIdx < bufSize; bufIdx++) {
                        switch(parserState) {
                        case 0: { // seeking '>'
                            if (buffer[bufIdx] == '>') {

                                that->frameIdxLock.Lock();
                                if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
                                that->frameIdx[frame++] = bufPos + bufIdx;
                                that->frameIdxEvent.Set();
                                that->frameIdxLock.Unlock();

                                parserState = 1;
                            }
                        } break;
                        case 1: { // whitespaces before particle number
                            if (!vislib::CharTraitsA::IsSpace(buffer[bufIdx])) {
                                *token.AllocateBuffer(1) = buffer[bufIdx];
                                parserState = 2;
                            }
                        } break;
                        case 2: { // particle number
                            if (!vislib::CharTraitsA::IsDigit(buffer[bufIdx])) {
                                framePartCnt = vislib::CharTraitsA::ParseUInt64(token);
                                if ((framePartCnt != that->dataHeader.GetParticleCount()) && (that->dataHeader.GetParticleCount() != 0)) {
                                    throw new vislib::Exception("Particle count changed between frames even the header already defined the count", __FILE__, __LINE__);
                                }
                                partIdx = 0;
                                parserState = (framePartCnt > 0) ? 
                                    ((buffer[bufIdx] == 0x0A) ? 4 : 3)
                                    : 0;
                            } else {
                                token.Append(buffer[bufIdx]);
                            }
                        } break;
                        case 3: { // linebreak after particle number
                            if (buffer[bufIdx] == 0x0A) parserState = 4;
                        } break;
                        case 4: { // particle line
                            if (buffer[bufIdx] == 0x0A) {
                                partIdx++;
                                if (partIdx == framePartCnt) {
                                    parserState = 0;
                                }
                            }
                        } break;
                        }
                    }
                    if (f.IsEOF()) {
                        if (frame == that->dataHeader.GetTimeCount()) {
                            that->frameIdxLock.Lock();
                            if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
                            that->frameIdx[frame] = f.Tell();
                            that->frameIdxEvent.Set();
                            that->frameIdxLock.Unlock();
                        } else {
                            that->frameIdxLock.Lock();
                            for (; frame <= frameCount; frame++) {
                                that->frameIdx[frame] = ULLONG_MAX;
                            }
                            that->frameIdxEvent.Set();
                            that->frameIdxLock.Unlock();
                            break;
                        }
                    }
                }

            }

        }

        that->frameIdxLock.Lock();
        if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
        UINT64 begin = that->frameIdx[0];
        UINT64 end = that->frameIdx[frameCount];
        that->frameIdxLock.Unlock();
        if ((begin == 0) || (end == 0)) {
            throw vislib::Exception("Frame index incomplete", __FILE__, __LINE__);
        } else {
            vislib::sys::Log::DefaultLog.WriteInfo(50, "Frame index of %u frames completed with ~%u bytes per frame",
                static_cast<unsigned int>(frameCount),
                static_cast<unsigned int>((end - begin) / frameCount));

#if defined(DEBUG) || defined(_DEBUG)
            //that->frameIdxLock.Lock();
            //if (that->frameIdx == NULL) { that->frameIdxLock.Unlock(); throw vislib::Exception("aborted", __FILE__, __LINE__); }
            //for (unsigned int i = 0; i <= frameCount; i++) {
            //    vislib::sys::Log::DefaultLog.WriteInfo(250, "    frame %u: %lu", i, that->frameIdx[i]);
            //}
            //that->frameIdxLock.Unlock();
#endif /* DEBUG || _DEBUG */

        }

    } catch(vislib::Exception e) {
        // sort of failed ...
        that->frameIdxLock.Lock();
        if (that->frameIdx != NULL) that->frameIdx[0] = 0;
        vislib::sys::Log::DefaultLog.WriteError("Failed to generated frame index: %s [%s:%d]",
            e.GetMsgA(), e.GetFile(), e.GetLine());
        that->frameIdxLock.Unlock();
        that->frameIdxEvent.Set();

    } catch(...) {
        // sort of failed ...
        that->frameIdxLock.Lock();
        if (that->frameIdx != NULL) that->frameIdx[0] = 0;
        vislib::sys::Log::DefaultLog.WriteError("Failed to generated frame index: unexpected exception");
        that->frameIdxLock.Unlock();
        that->frameIdxEvent.Set();
    }

    delete[] typeSizes;
    delete[] buffer;
    f.Close();

    return 0;
}


/*
 * MMSPDDataSource::clearData
 */
void MMSPDDataSource::clearData(void) {
    if (this->frameIdxThread.IsRunning()) {
        this->frameIdxLock.Lock();
        ARY_SAFE_DELETE(this->frameIdx);
        this->frameIdxLock.Unlock();
        this->frameIdxThread.Join();
    }

    this->frameIdxLock.Lock();
    this->resetFrameCache();
    this->dataHeader.SetParticleCount(0);
    this->dataHeader.SetTimeCount(1);
    this->dataHeader.BoundingBox().Set(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
    this->dataHeader.Types().Clear();
    this->clipbox = this->dataHeader.GetBoundingBox();
    ARY_SAFE_DELETE(this->frameIdx);
    this->setFrameCount(1);
    this->initFrameCache(1);
    this->frameIdxLock.Unlock();
}


/*
 * MMSPDDataSource::filenameChanged
 */
bool MMSPDDataSource::filenameChanged(core::param::ParamSlot& slot) {
    using vislib::sys::Log;
    using vislib::sys::File;
    this->clearData();
    this->resetFrameCache();

    if (this->file == NULL) {
        this->file = new vislib::sys::File();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<core::param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<core::param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        this->GetCoreInstance()->Log().WriteMsg(Log::LEVEL_ERROR, "Unable to open MMSPD-File \"%s\".", vislib::StringA(
            this->filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

#define _ERROR_OUT(MSG) { Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, MSG); \
        SAFE_DELETE(this->file); \
        this->clearData(); \
        return true; }
#define _ASSERT_READFILE(BUFFER, BUFFERSIZE) { if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
        _ERROR_OUT("Unable to read MMSPD file: seems truncated"); \
    } }
#define _ASSERT_READSTRINGBINARY(STRING) { (STRING).Clear(); while(true) { char c; _ASSERT_READFILE(&c, 1) if (c == 0) break; (STRING).Append(c); } }
#define _ASSERT_READLINE(STRING) { (STRING).Clear(); while(true) { char c; _ASSERT_READFILE(&c, 1) (STRING).Append(c); if (c == 0x0A) break; } }

    // reading format marker
    BYTE headerID[9];
    _ASSERT_READFILE(headerID, 9);
    bool jmpBk, text, unicode, bigEndian = false;
    
    // scharnkn: I'm pretty sure the code below does not work due to the short 
    // circuit evaluation of conditional statements in C++. The execution stops
    // once the result of the conditional statement is clear, meaning for
    // headerID == "MMSPDb", jmpBk is not set to true and consequently the file
    // loading will fail.
    //if ((text = (::memcmp(headerID, "MMSPDb", 6) != 0))
    //        && (unicode = (::memcmp(headerID, "MMSPDa", 6) != 0))
    //        && (::memcmp(headerID, "MMSPDu", 6) != 0)
    //       && (jmpBk = (::memcmp(headerID, "\xEF\xBB\xBFMMSPDu", 9) != 0))) {
    //    _ERROR_OUT("MMSPD format marker not found");
    //}
    // FIX
    text = (::memcmp(headerID, "MMSPDb", 6) != 0);
    unicode = (::memcmp(headerID, "MMSPDa", 6) != 0);
    bool retVal = (::memcmp(headerID, "MMSPDu", 6) != 0);
    jmpBk = (::memcmp(headerID, "\xEF\xBB\xBFMMSPDu", 9) != 0);
    if (text && unicode && retVal && jmpBk) {
        _ERROR_OUT("MMSPD format marker not found");
    }

    if (jmpBk) {
        this->file->Seek(-3, vislib::sys::File::CURRENT);
    }

    // read format version information
    vislib::VersionNumber version;
    if (text) {
        // the is ultimatively slow, but, it is okey here
        char c;
        vislib::StringA verStr;
        do {
            _ASSERT_READFILE(&c, 1);
            verStr.Append(c);
        } while (c != 0x0A);
        verStr.TrimSpaces();
        version.Parse(verStr);

        if (version != vislib::VersionNumber(1, 0)) {
            vislib::StringA msg;
            msg.Format("Version %s found. Supporting only version 1.0", version.ToStringA().PeekBuffer());
            _ERROR_OUT(msg);
        }

    } else {
        unsigned char buf[14];
        _ASSERT_READFILE(&buf, 14);
        if ((buf[0] != 0x00) || (buf[1] != 0xFF)) {
            _ERROR_OUT("MMSPD file format marker sequence broken @1");
        }
        unsigned int endianessTest;
        ::memcpy(&endianessTest, &buf[2], 4);
        if (endianessTest == 0x78563412) { // which is 2018915346u as specified
            bigEndian = false;
        } else if (endianessTest == 0x12345678) {
            bigEndian = true;
        } else {
            _ERROR_OUT("MMSPD file format marker sequence broken @2");
        }
        unsigned short majorVer, minorVer;
        if (bigEndian) {
            vislib::Swap(buf[6], buf[7]);
            vislib::Swap(buf[8], buf[9]);
        }
        ::memcpy(&majorVer, &buf[6], 2);
        ::memcpy(&minorVer, &buf[8], 2);
        if ((majorVer != 1) && (minorVer != 0)) {
            vislib::StringA msg;
            msg.Format("Version %d.%d found. Supporting only version 1.0", static_cast<int>(majorVer), static_cast<int>(minorVer));
            _ERROR_OUT(msg);
        }
        version.Set(majorVer, minorVer);
        for (int i = 10; i < 14; i++) {
            if (buf[i] < 128) {
                vislib::sys::Log::DefaultLog.WriteWarn("MMSPD file format marker binary guard byte %d illegal", (i - 9));
            }
        }

    }
    // file format marker successfully read
    // file pointer is a the start of the next line/data block

    // reading header line and particle types definitions
    if (text) {
        // reading header line
        vislib::StringA line;
        _ASSERT_READLINE(line);
        if (unicode) {
            vislib::StringW uniLine;
            if (!vislib::UTF8Encoder::Decode(uniLine, line)) {
                _ERROR_OUT("Failed to decode UTF8 header line");
            }
            line = uniLine;
        }
        line.TrimSpaces();
        vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(line, ' ', true));
        if (tokens.Count() < 10) {
            _ERROR_OUT("Header line incomplete");
        } else if (tokens.Count() > 10) {
            vislib::sys::Log::DefaultLog.WriteWarn("Trailing information on header line will be ignored");
        }

        const char *fieldName = "unknown";
        try {
            fieldName = "hasIDs";
            bool hasIDs = vislib::CharTraitsA::ParseBool(tokens[0]);
            fieldName = "minX";
            double minX = vislib::CharTraitsA::ParseDouble(tokens[1]);
            fieldName = "minY";
            double minY = vislib::CharTraitsA::ParseDouble(tokens[2]);
            fieldName = "minZ";
            double minZ = vislib::CharTraitsA::ParseDouble(tokens[3]);
            fieldName = "maxX";
            double maxX = vislib::CharTraitsA::ParseDouble(tokens[4]);
            fieldName = "maxY";
            double maxY = vislib::CharTraitsA::ParseDouble(tokens[5]);
            fieldName = "maxZ";
            double maxZ = vislib::CharTraitsA::ParseDouble(tokens[6]);
            fieldName = "timeCount";
            UINT32 timeCount = static_cast<UINT32>(vislib::CharTraitsA::ParseUInt64(tokens[7]));
            fieldName = "typeCount";
            UINT32 typeCount = static_cast<UINT32>(vislib::CharTraitsA::ParseUInt64(tokens[8]));
            fieldName = "partCount";
            UINT64 partCount = vislib::CharTraitsA::ParseUInt64(tokens[9]);

            this->dataHeader.BoundingBox().Set(minX, minY, minZ, maxX, maxY, maxZ);
            this->dataHeader.SetHasIDs(hasIDs);
            this->dataHeader.SetParticleCount(partCount);
            this->dataHeader.SetTimeCount(timeCount);
            this->dataHeader.Types().SetCount(typeCount);

        } catch(...) {
            vislib::StringA msg;
            msg.Format("Failed to parse header line file \"%s\"", fieldName);
            _ERROR_OUT(fieldName);
        }

        // reading particle types
        // Note: This is the only place where 'unicode' is really relevant!
        for (UINT32 typeIdx = 0; typeIdx < this->dataHeader.Types().Count(); typeIdx++) {
            MMSPDHeader::TypeDefinition &type = this->dataHeader.Types()[typeIdx];
            UINT32 constFieldCnt, fieldCnt;
            vislib::StringA str;

            _ASSERT_READLINE(line);
            if (unicode) {
                vislib::StringW uniLine;
                if (!vislib::UTF8Encoder::Decode(uniLine, line)) {
                    vislib::StringA msg;
                    msg.Format("Failed to decode UTF8 particle line %d", static_cast<int>(typeIdx));
                    _ERROR_OUT(msg);
                }
                line = uniLine;
            }
            line.TrimSpaces();
            tokens = vislib::StringTokeniserA::Split(line, ' ', true);
            if (tokens.Count() < 3) {
                vislib::StringA msg;
                msg.Format("Particle line %d incomplete", static_cast<int>(typeIdx));
                _ERROR_OUT(msg);
            }

            try {
                constFieldCnt = static_cast<UINT32>(vislib::CharTraitsA::ParseUInt64(tokens[1]));
            } catch(...) {
                vislib::StringA msg;
                msg.Format("Failed to parse fixFieldCount of particle line %d", static_cast<int>(typeIdx));
                _ERROR_OUT(msg);
            }

            try {
                fieldCnt = static_cast<UINT32>(vislib::CharTraitsA::ParseUInt64(tokens[2]));
            } catch(...) {
                vislib::StringA msg;
                msg.Format("Failed to parse varFieldCount of particle line %d", static_cast<int>(typeIdx));
                _ERROR_OUT(msg);
            }

            if (tokens.Count() < 3 + constFieldCnt * 3 + fieldCnt * 2) {
                vislib::StringA msg;
                msg.Format("Particle line %d incomplete", static_cast<int>(typeIdx));
                _ERROR_OUT(msg);
            } else if (tokens.Count() > 3 + constFieldCnt * 3 + fieldCnt * 2) {
                vislib::sys::Log::DefaultLog.WriteWarn("Trailing information on particle line %d will be ignored", static_cast<int>(typeIdx));
            }

            type.SetBaseType(tokens[0]);
            type.ConstFields().SetCount(constFieldCnt);
            type.Fields().SetCount(fieldCnt);
            int pos = 3;
            
            for (UINT32 fieldIdx = 0; fieldIdx < constFieldCnt; fieldIdx++, pos += 3) {
                MMSPDHeader::ConstField &field = type.ConstFields()[fieldIdx];

                if (tokens[pos].Equals("id")) _ERROR_OUT("Field \"id\" is reserved for internal use and must not be used!");
                if (tokens[pos].Equals("type")) _ERROR_OUT("Field \"type\" is reserved for internal use and must not be used!");
                if (tokens[pos + 1].Equals("b") || tokens[pos + 1].Equals("byte", false)) field.SetType(MMSPDHeader::Field::TYPE_BYTE);
                else if (tokens[pos + 1].Equals("f") || tokens[pos + 1].Equals("float", false)) field.SetType(MMSPDHeader::Field::TYPE_FLOAT);
                else if (tokens[pos + 1].Equals("d") || tokens[pos + 1].Equals("double", false)) field.SetType(MMSPDHeader::Field::TYPE_DOUBLE);
                else {
                    str.Format("Type \"%s\" of field \"%d\" of type definition \"%d\" is unknown",
                        tokens[pos + 1].PeekBuffer(), static_cast<int>(fieldIdx), static_cast<int>(typeIdx));
                   _ERROR_OUT(str);
                }
                field.SetName(tokens[pos]);
                try {
                    switch (field.GetType()) {
                        case MMSPDHeader::Field::TYPE_BYTE: {
                            int i = vislib::CharTraitsA::ParseInt(tokens[pos + 2]);
                            if ((i < 0) || (i > 255)) {
                                str.Format("Byte value of field \"%s\" of type %d out of range\n",
                                    tokens[pos].PeekBuffer(), static_cast<int>(typeIdx));
                               _ERROR_OUT(str);
                            }
                            field.SetByte(static_cast<unsigned char>(i));
                        } break;
                        case MMSPDHeader::Field::TYPE_FLOAT: {
                            field.SetFloat(static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[pos + 2])));
                        } break;
                        case MMSPDHeader::Field::TYPE_DOUBLE: {
                            field.SetDouble(vislib::CharTraitsA::ParseDouble(tokens[pos + 2]));
                        } break;
                        default: _ERROR_OUT("Internal Error!");
                    }
                } catch(...) {
                    str.Format("Failed to parse value for field \"%s\" of type %d\n",
                        tokens[pos].PeekBuffer(), static_cast<int>(typeIdx));
                   _ERROR_OUT(str);
                }
            }

            for (UINT32 fieldIdx = 0; fieldIdx < fieldCnt; fieldIdx++, pos += 2) {
                MMSPDHeader::Field &field = type.Fields()[fieldIdx];

                if (tokens[pos].Equals("id")) _ERROR_OUT("Field \"id\" is reserved for internal use and must not be used!");
                if (tokens[pos].Equals("type")) _ERROR_OUT("Field \"type\" is reserved for internal use and must not be used!");
                if (tokens[pos + 1].Equals("b") || tokens[pos + 1].Equals("byte", false)) field.SetType(MMSPDHeader::Field::TYPE_BYTE);
                else if (tokens[pos + 1].Equals("f") || tokens[pos + 1].Equals("float", false)) field.SetType(MMSPDHeader::Field::TYPE_FLOAT);
                else if (tokens[pos + 1].Equals("d") || tokens[pos + 1].Equals("double", false)) field.SetType(MMSPDHeader::Field::TYPE_DOUBLE);
                else {
                    str.Format("Type \"%s\" of field \"%d\" of type definition \"%d\" is unknown",
                        tokens[pos + 1].PeekBuffer(), static_cast<int>(fieldIdx), static_cast<int>(typeIdx));
                   _ERROR_OUT(str);
                }
                field.SetName(tokens[pos]);
            }

        }

    } else {
        // reading header line
        unsigned char hasIDs;
        double minX, minY, minZ, maxX, maxY, maxZ;
        UINT32 timeCount;
        UINT32 typeCount;
        UINT64 partCount;

        _ASSERT_READFILE(&hasIDs, 1);
        _ASSERT_READFILE(&minX, 8);
        _ASSERT_READFILE(&minY, 8);
        _ASSERT_READFILE(&minZ, 8);
        _ASSERT_READFILE(&maxX, 8);
        _ASSERT_READFILE(&maxY, 8);
        _ASSERT_READFILE(&maxZ, 8);
        _ASSERT_READFILE(&timeCount, 4);
        _ASSERT_READFILE(&typeCount, 4);
        _ASSERT_READFILE(&partCount, 8);
        

        // now I am confident enough to start setting data
        this->dataHeader.BoundingBox().Set(minX, minY, minZ, maxX, maxY, maxZ);
        this->dataHeader.SetHasIDs(hasIDs != 0);
        this->dataHeader.SetParticleCount(partCount);
        this->dataHeader.SetTimeCount(timeCount);
        this->dataHeader.Types().SetCount(typeCount);

        // reading particle types
        for (UINT32 typeIdx = 0; typeIdx < typeCount; typeIdx++) {
            MMSPDHeader::TypeDefinition &type = this->dataHeader.Types()[typeIdx];
            vislib::StringA str;
            UINT32 constFieldCnt, fieldCnt;

            _ASSERT_READSTRINGBINARY(str);
            _ASSERT_READFILE(&constFieldCnt, 4);
            if (bigEndian) {
                unsigned char *fac = reinterpret_cast<unsigned char*>(&constFieldCnt);
                vislib::Swap(fac[0], fac[3]);
                vislib::Swap(fac[1], fac[2]);
            }
            _ASSERT_READFILE(&fieldCnt, 4);
            if (bigEndian) {
                unsigned char *fac = reinterpret_cast<unsigned char*>(&fieldCnt);
                vislib::Swap(fac[0], fac[3]);
                vislib::Swap(fac[1], fac[2]);
            }

            type.SetBaseType(str);
            type.ConstFields().SetCount(constFieldCnt);
            type.Fields().SetCount(fieldCnt);

            for (UINT32 fieldIdx = 0; fieldIdx < constFieldCnt; fieldIdx++) {
                vislib::StringA typeStr;
                MMSPDHeader::ConstField &field = type.ConstFields()[fieldIdx];

                _ASSERT_READSTRINGBINARY(str);
                if (str.Equals("id")) _ERROR_OUT("Field \"id\" is reserved for internal use and must not be used!");
                if (str.Equals("type")) _ERROR_OUT("Field \"type\" is reserved for internal use and must not be used!");
                _ASSERT_READSTRINGBINARY(typeStr);
                if (typeStr.Equals("b") || typeStr.Equals("byte", false)) field.SetType(MMSPDHeader::Field::TYPE_BYTE);
                else if (typeStr.Equals("f") || typeStr.Equals("float", false)) field.SetType(MMSPDHeader::Field::TYPE_FLOAT);
                else if (typeStr.Equals("d") || typeStr.Equals("double", false)) field.SetType(MMSPDHeader::Field::TYPE_DOUBLE);
                else {
                    str.Format("Type \"%s\" of field \"%d\" of type definition \"%d\" is unknown",
                        typeStr.PeekBuffer(), static_cast<int>(fieldIdx), static_cast<int>(typeIdx));
                   _ERROR_OUT(str);
                }
                field.SetName(str);
                switch (field.GetType()) {
                    case MMSPDHeader::Field::TYPE_BYTE: {
                        unsigned char b;
                        _ASSERT_READFILE(&b, 1);
                        field.SetByte(b);
                    } break;
                    case MMSPDHeader::Field::TYPE_FLOAT: {
                        float f;
                        _ASSERT_READFILE(&f, 4);
                        if (bigEndian) {
                            unsigned char *fac = reinterpret_cast<unsigned char*>(&f);
                            vislib::Swap(fac[0], fac[3]);
                            vislib::Swap(fac[1], fac[2]);
                        }
                        field.SetFloat(f);
                    } break;
                    case MMSPDHeader::Field::TYPE_DOUBLE: {
                        double d;
                        _ASSERT_READFILE(&d, 8);
                        if (bigEndian) {
                            unsigned char *fac = reinterpret_cast<unsigned char*>(&d);
                            vislib::Swap(fac[0], fac[7]);
                            vislib::Swap(fac[1], fac[6]);
                            vislib::Swap(fac[2], fac[5]);
                            vislib::Swap(fac[3], fac[4]);
                        }
                        field.SetDouble(d);
                    } break;
                    default: _ERROR_OUT("Internal Error!");
                }
            }

            for (UINT32 fieldIdx = 0; fieldIdx < fieldCnt; fieldIdx++) {
                vislib::StringA typeStr;
                MMSPDHeader::Field &field = type.Fields()[fieldIdx];

                _ASSERT_READSTRINGBINARY(str);
                if (str.Equals("id")) _ERROR_OUT("Field \"id\" is reserved for internal use and must not be used!");
                if (str.Equals("type")) _ERROR_OUT("Field \"type\" is reserved for internal use and must not be used!");
                _ASSERT_READSTRINGBINARY(typeStr);
                if (typeStr.Equals("b") || typeStr.Equals("byte", false)) field.SetType(MMSPDHeader::Field::TYPE_BYTE);
                else if (typeStr.Equals("f") || typeStr.Equals("float", false)) field.SetType(MMSPDHeader::Field::TYPE_FLOAT);
                else if (typeStr.Equals("d") || typeStr.Equals("double", false)) field.SetType(MMSPDHeader::Field::TYPE_DOUBLE);
                else {
                    str.Format("Type \"%s\" of field \"%d\" of type definition \"%d\" is unknown",
                        typeStr.PeekBuffer(), static_cast<int>(fieldIdx), static_cast<int>(typeIdx));
                   _ERROR_OUT(str);
                }
                field.SetName(str);
            }

        }

    }
    this->isBinaryFile = !text;
    this->isBigEndian = bigEndian;
    if (this->dataHeader.GetTimeCount() == 0) {
        _ERROR_OUT("The data file seems to contain no data (no time frames)");
    }

    // reading frames
     
    //  index generation and size estimation
    this->frameIdx = new UINT64[this->dataHeader.GetTimeCount() + 1];
    ZeroMemory(this->frameIdx, sizeof(UINT64) * (this->dataHeader.GetTimeCount() + 1));
    this->frameIdx[0] = static_cast<UINT64>(this->file->Tell());

    if (this->dataHeader.GetTimeCount() == 1) {
        this->frameIdx[1] = static_cast<UINT64>(this->file->GetSize());
        this->setFrameCount(1);
        this->initFrameCache(1);
    } else {
        this->setFrameCount(this->dataHeader.GetTimeCount());
        this->frameIdxThread.Start(static_cast<void*>(this));
        // this->frameIdxThread.Join(); // Use this pause the main thread for debugging

        // estimate data set frame memory foot print
        SIZE_T dataSizeInMem = 1024 * 1024 * 10;
        double maxTypeGrowth = 0.0;
        SIZE_T maxFields = 0;

        for (SIZE_T i = 0; i < this->dataHeader.GetTypes().Count(); i++) {
            SIZE_T fs = 0;
            SIZE_T ms = 0;
            const MMSPDHeader::TypeDefinition &type = this->dataHeader.GetTypes()[i];
            if (maxFields < type.GetFields().Count()) {
                maxFields = type.GetFields().Count();
            }
            for (SIZE_T j = 0; j < type.GetFields().Count(); j++) {
                ms += sizeof(float);
                switch (type.GetFields()[j].GetType()) {
                case MMSPDHeader::Field::TYPE_BYTE: fs += 1; break;
                case MMSPDHeader::Field::TYPE_FLOAT: fs += 4; break;
                case MMSPDHeader::Field::TYPE_DOUBLE: fs += 8; break;
                default: fs += 0; break;
                }
            }
            double typeGrowth = static_cast<double>(ms) / static_cast<double>(fs);
            if (typeGrowth > maxTypeGrowth) {
                maxTypeGrowth = typeGrowth;
            }
        }
        if (maxTypeGrowth < 0.4) {
            // I really don't think so!
            maxTypeGrowth = 0.4;
        }

        maxTypeGrowth *= static_cast<double>(CACHE_FRAME_FACTOR);

        UINT64 fLs[4];
        do {
            this->frameIdxLock.Lock();
            fLs[0] = this->frameIdx[0];
            fLs[1] = this->frameIdx[1];
            fLs[2] = this->frameIdx[2];
            if (this->dataHeader.GetTimeCount() > 2) fLs[3] = this->frameIdx[3]; else fLs[3] = fLs[2];
            this->frameIdxLock.Unlock();

            if (fLs[0] == 0) {
                // aborted with error
                _ERROR_OUT("Unable to load first frame (Frame index generation aborted)");
            }
            if ((fLs[1] == 0) || (fLs[2] == 0) || (fLs[3] == 0)) {
                this->frameIdxEvent.Reset();
                this->frameIdxEvent.Wait();
            }

        } while ((fLs[1] == 0) || (fLs[2] == 0) || (fLs[3] == 0));

        if (this->isBinaryFile) {
            double maxFrameSize = static_cast<double>(vislib::math::Max(
                vislib::math::Max(fLs[1] - fLs[0], fLs[2] - fLs[1]), fLs[3] - fLs[2]));
            dataSizeInMem = static_cast<SIZE_T>(maxFrameSize * maxTypeGrowth);

        } else {
            UINT64 pcnt = this->dataHeader.GetParticleCount();

            for (int fidx = 0; fidx < ((fLs[3] == fLs[2]) ? 2 : 3); fidx++) {
                this->file->Seek(static_cast<vislib::sys::File::FileOffset>(fLs[fidx]));
                vislib::StringA ln = vislib::sys::ReadLineFromFileA(*this->file);
                if (ln.IsEmpty()) {
                    _ERROR_OUT("Frame table broken (1)");
                }
                if (ln[0] != '>') {
                    _ERROR_OUT("Frame table broken (2)");
                }
                ln = ln.Substring(1);
                ln.TrimSpacesBegin();
                try {
                    UINT64 pc = vislib::CharTraitsA::ParseUInt64(ln);
                    pcnt = vislib::math::Max(pcnt, pc);
                } catch(...) {
                    _ERROR_OUT("Frame marker error");
                }
            }

            dataSizeInMem = static_cast<SIZE_T>(pcnt * maxFields * sizeof(float));

        }

        if (dataSizeInMem == 0) {
            _ERROR_OUT("Unable to load first frame (no data)");
        }

        //dataSizeInMem += static_cast<UINT64>(static_cast<double>(dataSizeInMem)
        //    * (static_cast<double>(CACHE_FRAME_FACTOR) - 1.0));

        UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
        unsigned int cacheSize = static_cast<unsigned int>(mem / dataSizeInMem);
        if (cacheSize > this->dataHeader.GetTimeCount()) {
            cacheSize = this->dataHeader.GetTimeCount();
        }

        if (cacheSize > CACHE_SIZE_MAX) {
            vislib::sys::Log::DefaultLog.WriteInfo("Frame cache size %u requested limited to %d",
                cacheSize, static_cast<int>(CACHE_SIZE_MAX));
            cacheSize = CACHE_SIZE_MAX;
        }
        if (cacheSize < CACHE_SIZE_MIN) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Frame cache size forced to %d. Calculated size was %u.\n",
                static_cast<int>(CACHE_SIZE_MIN), cacheSize);
            cacheSize = CACHE_SIZE_MIN;
        } else {
            vislib::sys::Log::DefaultLog.WriteInfo("Frame cache size set to %u.\n",
                cacheSize);
        }
        this->initFrameCache(cacheSize);
    }

    this->dataHash++;

#undef _ASSERT_READLINE
#undef _ASSERT_READSTRINGBINARY
#undef _ASSERT_READFILE
#undef _ERROR_OUT

    return true;
}


/*
 * MMSPDDataSource::getDataCallback
 */
bool MMSPDDataSource::getDataCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    Frame *f = NULL;
    if (c2 != NULL) {
        //vislib::sys::Log::DefaultLog.WriteInfo("MMSPDDataSource: got a request for frame %u", c2->FrameID());
        do {
            f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID()));
        } while (c2->IsFrameForced() && f->FrameNumber() != c2->FrameID() && (f->Unlock(), true)); // either the frame is irrelevant, or the frame is okay, or we need to unlock!
        if (f == NULL) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        //vislib::sys::Log::DefaultLog.WriteInfo("MMSPDDataSource: providing frame %u", f->FrameNumber());
        c2->SetDataHash(this->dataHash);
        f->SetData(*c2, this->dataHeader);
    }

    return true;
}


/*
 * MMSPDDataSource::getDirDataCallback
 */
bool MMSPDDataSource::getDirDataCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall* c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == nullptr) return false;

    Frame *f = nullptr;
    if (c2 != nullptr) {
        //vislib::sys::Log::DefaultLog.WriteInfo("MMSPDDataSource: got a request for frame %u", c2->FrameID());
        do {
            f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID()));
        } while (c2->IsFrameForced() && f->FrameNumber() != c2->FrameID() && (f->Unlock(), true)); // either the frame is irrelevant, or the frame is okay, or we need to unlock!
        if (f == nullptr) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        //vislib::sys::Log::DefaultLog.WriteInfo("MMSPDDataSource: providing frame %u", f->FrameNumber());
        c2->SetDataHash(this->dataHash);
        f->SetDirData(*c2, this->dataHeader);
    }

    return true;
}


/*
 * MMSPDDataSource::getExtentCallback
 */
bool MMSPDDataSource::getExtentCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetFrameCount(vislib::math::Max(1u, this->dataHeader.GetTimeCount()));
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->dataHeader.GetBoundingBox());
        //c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->dataHeader.GetBoundingBox());
        c2->SetDataHash((this->CacheSize() > 0) ? this->dataHash : 0);
        c2->SetUnlocker(NULL);
        return true;
    }

    core::moldyn::MultiParticleDataCall* dirCall = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);

    if (dirCall != nullptr) {
        dirCall->SetFrameCount(vislib::math::Max(1u, this->dataHeader.GetTimeCount()));
        dirCall->AccessBoundingBoxes().Clear();
        dirCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->dataHeader.GetBoundingBox());
        //dirCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        dirCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->dataHeader.GetBoundingBox());
        dirCall->SetDataHash((this->CacheSize() > 0) ? this->dataHash : 0);
        dirCall->SetUnlocker(nullptr);

        return true;
    }

    return false;
}
