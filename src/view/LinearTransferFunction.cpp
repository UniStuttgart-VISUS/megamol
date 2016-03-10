/*
 * LinearTransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/LinearTransferFunction.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace core {
namespace view {

    static int InterColourComparer(const vislib::math::Vector<float, 5>& lhs, 
            const vislib::math::Vector<float, 5>& rhs) {
        if (lhs[4] >= rhs[4]) {
            if (rhs[4] + vislib::math::FLOAT_EPSILON >= lhs[4]) {
                return 0;
            } else {
                return 1;
            }
        } else {
            if (lhs[4] + vislib::math::FLOAT_EPSILON >= rhs[4]) {
                return 0;
            } else {
                return -1;
            }
        }
    }

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */


using namespace megamol::core;


/*
 * view::LinearTransferFunction::LinearTransferFunction
 */
view::LinearTransferFunction::LinearTransferFunction(void) : Module(),
        getTFSlot("gettransferfunction", "Provides the transfer function"),
        minColSlot("mincolour", "The colour for the minimum value"),
        maxColSlot("maxcolour", "The colour for the maximum value"),
        texSizeSlot("texsize", "The size of the texture to generate"),
        pathSlot("filepath", "path for serializing the TF"),
        loadTFSlot("loadTF", "trigger loading from file"),
        storeTFSlot("storeTF", "trigger saving to file"),
        texID(0), texSize(1),
        texFormat(CallGetTransferFunction::TEXTURE_FORMAT_RGB) {

    view::CallGetTransferFunctionDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0),
        &LinearTransferFunction::requestTF);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->minColSlot << new param::StringParam("blue");
    this->MakeSlotAvailable(&this->minColSlot);

    vislib::StringA t1, t2;
    for (SIZE_T i = 0; i < INTER_COLOUR_COUNT; i++) {
        t1.Format("enable%.2d", i + 1);
        t2.Format("Enables the intermediate colour %d", i + 1);
        this->interCols[i].enableSlot = new param::ParamSlot(t1, t2);
        this->interCols[i].enableSlot->SetParameter(new param::BoolParam(false));
        this->MakeSlotAvailable(this->interCols[i].enableSlot);

        t1.Format("colour%.2d", i + 1);
        t2.Format("The colour for the intermediate value no. %d", i + 1);
        this->interCols[i].colSlot = new param::ParamSlot(t1, t2);
        this->interCols[i].colSlot->SetParameter(new param::StringParam("Gray"));
        this->MakeSlotAvailable(this->interCols[i].colSlot);

        t1.Format("value%.2d", i + 1);
        t2.Format("The intermediate value no. %d", i + 1);
        this->interCols[i].valSlot = new param::ParamSlot(t1, t2);
        this->interCols[i].valSlot->SetParameter(new param::FloatParam(
            static_cast<float>(i + 1) / static_cast<float>(INTER_COLOUR_COUNT + 1),
            0.0f, 1.0f));
        this->MakeSlotAvailable(this->interCols[i].valSlot);
    }

    this->maxColSlot << new param::StringParam("red");
    this->MakeSlotAvailable(&this->maxColSlot);
    this->maxColSlot.ForceSetDirty();

    this->texSizeSlot << new param::IntParam(128, 2, 1024);
    this->MakeSlotAvailable(&this->texSizeSlot);

    this->pathSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->pathSlot);

    this->loadTFSlot << new param::ButtonParam();
    this->loadTFSlot.SetUpdateCallback(&LinearTransferFunction::loadTFPressed);
    this->MakeSlotAvailable(&this->loadTFSlot);

    this->storeTFSlot << new param::ButtonParam();
    this->storeTFSlot.SetUpdateCallback(&LinearTransferFunction::storeTFPressed);
    this->MakeSlotAvailable(&this->storeTFSlot);
}


/*
 * view::LinearTransferFunction::~LinearTransferFunction
 */
view::LinearTransferFunction::~LinearTransferFunction(void) {
    this->Release();
}


/*
 * view::LinearTransferFunction::create
 */
bool view::LinearTransferFunction::create(void) {
    // intentionally empty
    return true;
}


/*
 * view::LinearTransferFunction::release
 */
void view::LinearTransferFunction::release(void) {
    ::glDeleteTextures(1, &this->texID);
    this->texID = 0;
}


/*
 * view::LinearTransferFunction::loadTFPressed
 */
bool view::LinearTransferFunction::loadTFPressed(param::ParamSlot& param) {
    try {
        vislib::sys::BufferedFile inFile;
        if (!inFile.Open(pathSlot.Param<param::FilePathParam>()->Value(),
            vislib::sys::File::AccessMode::READ_ONLY,
            vislib::sys::File::ShareMode::SHARE_READ,
            vislib::sys::File::CreationMode::OPEN_ONLY)) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_WARN,
                "Unable to open TF file.");
            return false;
        }

        while (!inFile.IsEOF()) {
            vislib::StringA line = vislib::sys::ReadLineFromFileA(inFile);
            line.TrimSpaces();
            if (line.IsEmpty()) continue;
            if (line[0] == '#') continue;
            vislib::StringA::Size pos = line.Find('=');
            vislib::StringA name = line.Substring(0, pos);
            vislib::StringA value = line.Substring(pos + 1);
            Module::child_list_type::iterator ano_end = this->ChildList_End();
            for (Module::child_list_type::iterator ano_i = this->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
                std::shared_ptr<param::ParamSlot> p = std::dynamic_pointer_cast<param::ParamSlot>(*ano_i);
                if (p && p->Name().Equals(name)) {
                    p->Parameter()->ParseValue(value);
                }
            }
        }
        inFile.Close();
    } catch (...) {

    }

    return true;
}


/*
 * view::LinearTransferFunction::requestTF
 */
bool view::LinearTransferFunction::requestTF(Call& call) {
    view::CallGetTransferFunction *cgtf = dynamic_cast<view::CallGetTransferFunction*>(&call);
    if (cgtf == NULL) return false;

    bool dirty = this->minColSlot.IsDirty() || this->maxColSlot.IsDirty() || this->texSizeSlot.IsDirty();
    for (SIZE_T i = 0; !dirty && (i < INTER_COLOUR_COUNT); i++) {
        dirty = this->interCols[i].enableSlot->IsDirty()
            || this->interCols[i].colSlot->IsDirty()
            || this->interCols[i].valSlot->IsDirty();
    }
    if ((this->texID == 0) || dirty) {
        this->minColSlot.ResetDirty();
        this->maxColSlot.ResetDirty();
        this->texSizeSlot.ResetDirty();
        for (SIZE_T i = 0; i < INTER_COLOUR_COUNT; i++) {
            this->interCols[i].enableSlot->ResetDirty();
            this->interCols[i].colSlot->ResetDirty();
            this->interCols[i].valSlot->ResetDirty();
        }

        bool t1de = (glIsEnabled(GL_TEXTURE_1D) == GL_TRUE);
        if (!t1de) glEnable(GL_TEXTURE_1D);
        if (this->texID == 0) glGenTextures(1, &this->texID);

        vislib::Array<vislib::math::Vector<float, 5> > cols;
        vislib::math::Vector<float, 5> cx1, cx2;
        bool validAlpha = false;
        if (utility::ColourParser::FromString(
                this->minColSlot.Param<param::StringParam>()->Value(),
                cx1[0], cx1[1], cx1[2], cx1[3])) {
            cx1[4] = 0.0f;
        } else {
            cx1[0] = cx1[1] = cx1[2] = cx1[3] = 0.0f;
            cx1[4] = 0.0f;
        }
        if (cx1[3] < 0.99f) validAlpha = true;
        cols.Add(cx1);
        if (utility::ColourParser::FromString(
                this->maxColSlot.Param<param::StringParam>()->Value(),
                cx1[0], cx1[1], cx1[2], cx1[3])) {
            cx1[4] = 1.0f;
        } else {
            cx1[0] = cx1[1] = cx1[2] = cx1[3] = 0.0f;
            cx1[4] = 1.0f;
        }
        if (cx1[3] < 0.99f) validAlpha = true;
        cols.Add(cx1);
        for (SIZE_T i = 0; i < INTER_COLOUR_COUNT; i++) {
            if (this->interCols[i].enableSlot->Param<param::BoolParam>()->Value()) {
                float val = this->interCols[i].valSlot->Param<param::FloatParam>()->Value();
                if (utility::ColourParser::FromString(
                        this->interCols[i].colSlot->Param<param::StringParam>()->Value(),
                        cx1[0], cx1[1], cx1[2], cx1[3])) {
                    cx1[4] = val;
                } else {
                    cx1[0] = cx1[1] = cx1[2] = cx1[3] = 0.0f;
                    cx1[4] = val;
                }
                if (cx1[3] < 0.99f) validAlpha = true;
                cols.Add(cx1);
            }
        }

        cols.Sort(&InterColourComparer);

        this->texSize = this->texSizeSlot.Param<param::IntParam>()->Value();
        float *tex = new float[4 * this->texSize];
        int p1, p2;

        cx2 = cols[0];
        p2 = 0;
        for (SIZE_T i = 1; i < cols.Count(); i++) {
            cx1 = cx2;
            p1 = p2;
            cx2 = cols[i];
            ASSERT(cx2[4] <= 1.0f + vislib::math::FLOAT_EPSILON);
            p2 = static_cast<int>(cx2[4] * static_cast<float>(this->texSize - 1));
            ASSERT(p2 < static_cast<int>(this->texSize));
            ASSERT(p2 >= p1);

            for (int p = p1; p <= p2; p++) {
                float al = static_cast<float>(p - p1) / static_cast<float>(p2 - p1);
                float be = 1.0f - al;

                tex[p * 4] = cx1[0] * be + cx2[0] * al;
                tex[p * 4 + 1] = cx1[1] * be + cx2[1] * al;
                tex[p * 4 + 2] = cx1[2] * be + cx2[2] * al;
                tex[p * 4 + 3] = cx1[3] * be + cx2[3] * al;
            }
        }

        GLint otid = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_1D, &otid);
        glBindTexture(GL_TEXTURE_1D, this->texID);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->texSize, 0, GL_RGBA, GL_FLOAT, tex);

        delete[] tex;

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);

        glBindTexture(GL_TEXTURE_1D, otid);

        if (!t1de) glDisable(GL_TEXTURE_1D);

        this->texFormat = validAlpha
            ? CallGetTransferFunction::TEXTURE_FORMAT_RGBA
            : CallGetTransferFunction::TEXTURE_FORMAT_RGB;

    }

    cgtf->SetTexture(this->texID, this->texSize, this->texFormat);

    return true;
}


/*
* view::LinearTransferFunction::storeTFPressed
*/
bool view::LinearTransferFunction::storeTFPressed(param::ParamSlot& param) {

    try {
        vislib::sys::BufferedFile outFile;
        if (!outFile.Open(pathSlot.Param<param::FilePathParam>()->Value(),
            vislib::sys::File::AccessMode::WRITE_ONLY,
            vislib::sys::File::ShareMode::SHARE_EXCLUSIVE,
            vislib::sys::File::CreationMode::CREATE_OVERWRITE)) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_WARN,
                "Unable to create TF file.");
        }

        Module::child_list_type::iterator ano_end = this->ChildList_End();
        for (Module::child_list_type::iterator ano_i = this->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
            std::shared_ptr<param::ParamSlot> p = std::dynamic_pointer_cast<param::ParamSlot>(*ano_i);
            if (p && !p->Name().Equals("filepath")) {
                writeParameterFileParameter(*p, outFile);
            }
        }
        outFile.Close();
    } catch (...) {

    }

    return true;
}


/*
* view::LinearTransferFunction::writeParameterFileParameter
*/
void view::LinearTransferFunction::writeParameterFileParameter(
    param::ParamSlot& param,
    vislib::sys::BufferedFile &outFile) {

    unsigned int len = 0;
    vislib::RawStorage store;
    param.Parameter()->Definition(store);

    if (::memcmp(store, "MMBUTN", 6) == 0) {
        outFile.Write("# ", 2);
    }

    vislib::sys::WriteFormattedLineToFile(outFile,
        "%s=%s\n", param.Name(), vislib::StringA(param.Parameter()->ValueString()));
}