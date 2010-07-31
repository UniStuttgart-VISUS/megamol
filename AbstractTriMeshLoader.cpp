/*
 * AbstractTriMeshLoader.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AbstractTriMeshLoader.h"
#include "param/FilePathParam.h"
#include "vislib/assert.h"
#include "vislib/Log.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * AbstractTriMeshLoader::AbstractTriMeshLoader
 */
AbstractTriMeshLoader::AbstractTriMeshLoader(void) : AbstractTriMeshDataSource(),
        filenameSlot("filename", "The path to the file to load") {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

}


/*
 * AbstractTriMeshLoader::~AbstractTriMeshLoader
 */
AbstractTriMeshLoader::~AbstractTriMeshLoader(void) {
    this->Release();
    ASSERT(this->objs.IsEmpty());
    ASSERT(this->mats.IsEmpty());
}


/*
 * AbstractTriMeshLoader::assertData
 */
void AbstractTriMeshLoader::assertData(void) {
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->objs.Clear();
        this->mats.Clear();
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        bool retval = false;
        try {
            retval = this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
        } catch(vislib::Exception ex) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unexpected exception: %s at (%s, %d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
            retval = false;
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unexpected exception: unkown exception\n");
            retval = false;
        }
        if (retval) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Loaded file \"%s\"",
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Failed to load file \"%s\"",
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
            // ensure there is no partial data
            this->objs.Clear();
            this->mats.Clear();
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        this->datahash++;
    }
}
