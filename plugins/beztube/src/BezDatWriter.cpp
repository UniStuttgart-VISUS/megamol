/*
 * BezDatWriter.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "BezDatWriter.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/VersionNumber.h"

using namespace megamol;
using namespace megamol::beztube;


/*
 * BezDatWriter::BezDatWriter
 */
BezDatWriter::BezDatWriter(void) : base(),
        fileTypeSlot("filetype", "Type of the file to write"), fileType(0) {

    core::param::EnumParam *fileTypes = new core::param::EnumParam(this->fileType);
    fileTypes->SetTypePair(0, "Binary (v2)");
    fileTypes->SetTypePair(1, "ASCII (v2)");
    this->fileTypeSlot << fileTypes;
    this->MakeSlotAvailable(&this->fileTypeSlot);

}


/*
 * BezDatWriter::~BezDatWriter
 */
BezDatWriter::~BezDatWriter(void) {
    this->Release();
}


/*
 * BezDatWriter::writeFileStart
 */
void BezDatWriter::writeFileStart(vislib::sys::File& file, core::misc::BezierCurvesListDataCall& data) {
    this->fileType = this->fileTypeSlot.Param<core::param::EnumParam>()->Value();

    file.Write("BezDat", 6);
    if (this->fileType == 0) {
        // binary version 2
        file.Write("B", 1);
        unsigned char header[] = {
            0x00, 0xff, // string end
            0xA1, 0xA5, 0xAA, // binary filler
            0x02, 0x00, 0x00, 0x00 // version number 2.0.0.0
        };
        file.Write(header, 9);
        unsigned int endienTest1 = 0x12345678;
        float endienTest2 = 3.141f;
        file.Write(&endienTest1, 4);
        file.Write(&endienTest2, 4);
        // file is now 24 bytes long

        unsigned int frameCnt = data.FrameCount();
        file.Write(&frameCnt, 4); // uint32 frame count

    } else if (this->fileType == 1) {
        // ascii version 2
        file.Write("A ", 2);
        vislib::VersionNumber vn(2, 0);
        vislib::StringA vns = vn.ToStringA();
        file.Write(vns.PeekBuffer(), vns.Length());
        file.Write("\n", 1);

    } else {
        throw vislib::Exception(__FILE__, __LINE__);
    }
}


/*
 * BezDatWriter::writeFileFrameData
 */
void BezDatWriter::writeFileFrameData(vislib::sys::File& file, unsigned int idx, core::misc::BezierCurvesListDataCall& data) {
    if (this->fileType == 0) {
        // binary version 2

        uint32_t frameLen = 4; // size for frameLen itself
        for (size_t i = 0; i < data.Count(); i++) {
            const core::misc::BezierCurvesListDataCall::Curves& curves = data.GetCurves()[i];
            unsigned int bpp;
            bool needGlobRad = false;
            bool needGlobCol = false;
            switch (curves.GetDataLayout()) {
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_NONE: continue; // skip
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 12; needGlobRad = true; needGlobCol = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 16; needGlobCol = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 15; needGlobRad = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 19; break;
            }
            frameLen += 1; // format
            if (data.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                frameLen += 4 * 6;
            }
            if (data.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                frameLen += 4 * 6;
            }
            if (needGlobRad) frameLen += 4;
            if (needGlobCol) frameLen += 3;
            frameLen += 4; // point count;
            frameLen += static_cast<uint32_t>(bpp * curves.GetDataPointCount()); // point data
            if ((idx == 0) || !data.HasStaticIndices()) {
                frameLen += 4; // index count;
                frameLen += static_cast<uint32_t>(4 * curves.GetIndexCount()); // index data
            }
        }

        file.Write(&frameLen, 4);
        for (size_t i = 0; i < data.Count(); i++) {
            const core::misc::BezierCurvesListDataCall::Curves& curves = data.GetCurves()[i];
            unsigned int bpp;
            unsigned char layout = static_cast<unsigned char>(curves.GetDataLayout());
            bool needGlobRad = false;
            bool needGlobCol = false;
            switch (curves.GetDataLayout()) {
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_NONE: continue; // skip
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 12; needGlobRad = true; needGlobCol = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 16; needGlobCol = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 15; needGlobRad = true; break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 19; break;
            }
            if (data.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                layout |= 128; // highest bit for bounding box
            }
            if (data.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                layout |= 64; // second highest bit for clip box
            }
            if (data.HasStaticIndices()) layout |= 32; // third highest bit for static index structure
            file.Write(&layout, 1);
            if (data.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                file.Write(data.AccessBoundingBoxes().ObjectSpaceBBox().PeekBounds(), 4 * 6);
            }
            if (data.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                file.Write(data.AccessBoundingBoxes().ObjectSpaceClipBox().PeekBounds(), 4 * 6);
            }
            if (needGlobRad) {
                float f = curves.GetGlobalRadius();
                file.Write(&f, 4);
            }
            if (needGlobCol) file.Write(curves.GetGlobalColour(), 3);

            unsigned int cnt = static_cast<unsigned int>(curves.GetDataPointCount());
            file.Write(&cnt, 4);
            file.Write(curves.GetData(), cnt * bpp);

            if ((idx == 0) || !data.HasStaticIndices()) {
                cnt = static_cast<unsigned int>(curves.GetIndexCount());
                file.Write(&cnt, 4);
                file.Write(curves.GetIndex(), cnt * 4);
            }
        }

    } else if (this->fileType == 1) {
        // ascii version 2
        vislib::StringA line;
        line.Format("FRAME %u\n", idx);
        if ((idx == 0) && (data.HasStaticIndices())) line.Insert(line.Length() - 1, " STATIC_INDICES");
        file.Write(line.PeekBuffer(), line.Length());

        if (data.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
            const vislib::math::Cuboid<float>& box = data.AccessBoundingBoxes().ObjectSpaceBBox();
            line.Format("BBOX %g %g %g %g %g %g\n",
                box.GetLeft(), box.GetBottom(), box.GetBack(),
                box.GetRight(), box.GetTop(), box.GetFront());
            file.Write(line.PeekBuffer(), line.Length());
        }

        if (data.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
            const vislib::math::Cuboid<float>& box = data.AccessBoundingBoxes().ObjectSpaceClipBox();
            line.Format("CBOX %g %g %g %g %g %g\n",
                box.GetLeft(), box.GetBottom(), box.GetBack(),
                box.GetRight(), box.GetTop(), box.GetFront());
            file.Write(line.PeekBuffer(), line.Length());
        }

        for (size_t i = 0; i < data.Count(); i++) {
            const core::misc::BezierCurvesListDataCall::Curves& curves = data.GetCurves()[i];
            line.Format("LIST ");
            bool needGlobRad = false;
            bool needGlobCol = false;
            bool hasRad = false;
            bool hasCol = false;
            unsigned int bpp;
            switch (curves.GetDataLayout()) {
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_NONE: continue; // skip
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F:
                line.Append("XYZ\n");
                needGlobRad = true;
                needGlobCol = true;
                bpp = 12;
                break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F:
                line.Append("XYZR\n");
                needGlobCol = true;
                hasRad = true;
                bpp = 16;
                break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B:
                line.Append("XYZcol\n");
                needGlobRad = true;
                hasCol = true;
                bpp = 15;
                break;
            case core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B:
                line.Append("XYZRcol\n");
                hasRad = true;
                hasCol = true;
                bpp = 19;
                break;
            }
            file.Write(line.PeekBuffer(), line.Length());
            if (vislib::math::IsEqual(curves.GetGlobalRadius(), 0.5f)) {
                needGlobRad = false;
            }
            if ((curves.GetGlobalColour()[0] == 127)
                    && (curves.GetGlobalColour()[1] == 127)
                    && (curves.GetGlobalColour()[2] == 127)) {
                needGlobCol = false;
            }
//#if defined(DEBUG) || defined(_DEBUG)
//            // force: DEBUG!
//            needGlobCol = true;
//            needGlobRad = true;
//#endif
            if (needGlobRad) {
                line.Format("GLOBRAD %g\n", curves.GetGlobalRadius());
                file.Write(line.PeekBuffer(), line.Length());
            }
            if (needGlobCol) {
                line.Format("GLOBCOL %u %u %u\n", curves.GetGlobalColour()[0], curves.GetGlobalColour()[1], curves.GetGlobalColour()[2]);
                file.Write(line.PeekBuffer(), line.Length());
            }

            for (size_t j = 0; j < curves.GetDataPointCount(); j++) {
                const float *pos = curves.GetDataAt<float>(j * bpp);
                const unsigned char *col = curves.GetDataAt<unsigned char>(j * bpp + (hasRad) ? 16: 12);
                vislib::StringA frag;
                line.Format("PT %g %g %g", pos[0], pos[1], pos[2]);
                if (hasRad) {
                    frag.Format(" %g", pos[3]);
                    line.Append(frag);
                }
                if (hasCol) {
                    frag.Format(" %u %u %u", col[0], col[1], col[2]);
                    line.Append(frag);
                }
                line.Append("\n");
                file.Write(line.PeekBuffer(), line.Length());
            }

            if ((idx == 0) || !data.HasStaticIndices()) {
                for (size_t j = 0; j < curves.GetIndexCount(); j += 4) {
                    const unsigned int* idx = curves.GetIndex() + j;
                    line.Format("BC %u %u %u %u\n", idx[0], idx[1], idx[2], idx[3]);
                    file.Write(line.PeekBuffer(), line.Length());
                }
            }

            file.Write("LISTEND\n", 8);
        }
        file.Write("FRAMEEND\n", 7);

    } else {
        throw vislib::Exception(__FILE__, __LINE__);
    }
}


/*
 * BezDatWriter::writeFileEnd
 */
void BezDatWriter::writeFileEnd(vislib::sys::File& file) {
    if (this->fileType == 0) {
        // binary version 2
        // intentionally empty, loader will have to detect truncted files anyway

    } else if (this->fileType == 1) {
        // ascii version 2
        file.Write("END\n", 4);

    } else {
        throw vislib::Exception(__FILE__, __LINE__);
    }
}
