/*
 * AbstractTriMeshLoader.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "AbstractTriMeshLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/assert.h"

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * AbstractTriMeshLoader::AbstractTriMeshLoader
 */
AbstractTriMeshLoader::AbstractTriMeshLoader()
        : AbstractTriMeshDataSource()
        , filenameSlot("filename", "The path to the file to load") {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);
}


/*
 * AbstractTriMeshLoader::~AbstractTriMeshLoader
 */
AbstractTriMeshLoader::~AbstractTriMeshLoader() {
    this->Release();
    ASSERT(this->objs.IsEmpty());
    ASSERT(this->mats.IsEmpty());
}


/*
 * AbstractTriMeshLoader::assertData
 */
void AbstractTriMeshLoader::assertData() {
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->objs.Clear();
        this->mats.Clear();
        this->lines.clear();
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        bool retval = false;
        try {
            retval =
                this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_string().c_str());
        } catch (vislib::Exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unexpected exception: %s at (%s, %d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
            retval = false;
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Unexpected exception: unkown exception\n");
            retval = false;
        }
        if (retval) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("Loaded file \"%s\"",
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_string().c_str());
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to load file \"%s\"",
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_string().c_str());
            // ensure there is no partial data
            this->objs.Clear();
            this->mats.Clear();
            this->lines.clear();
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        this->datahash++;
    }
}
