/*
 * FEMMaterialDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "FEMMaterialDataSource.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "mesh/CallGPUMaterialData.h"

namespace megamol {
namespace archvis {

FEMMaterialDataSource::FEMMaterialDataSource()
    : m_btf_filename_slot("BTF filename", "The name of the btf file to load")
    , m_transferFunction_slot("gettransferfunction", "Connects to the transfer function module") {
    this->m_btf_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_btf_filename_slot);

    this->m_transferFunction_slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transferFunction_slot);
}

FEMMaterialDataSource::~FEMMaterialDataSource() {}


bool FEMMaterialDataSource::create() { return true; }

bool megamol::archvis::FEMMaterialDataSource::getDataCallback(core::Call& caller) {
    mesh::CallGPUMaterialData* mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    if (mtl_call == NULL) return false;

    core::view::CallGetTransferFunction* tf_call =
        this->m_transferFunction_slot.CallAs<core::view::CallGetTransferFunction>();
    if (tf_call == nullptr) return false;
    if (!((*tf_call)())) return false;

    auto tf_texture_name = tf_call->OpenGLTexture();

    // clear update?

    if (this->m_btf_filename_slot.IsDirty()) {
        m_btf_filename_slot.ResetDirty();

        auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

        m_gpu_materials->clearMaterials();

        m_gpu_materials->addMaterial(this->instance(), filename, {tf_texture_name});
    }

    if (tf_call->isDirty()) {
        if (!m_gpu_materials->getMaterials().empty()) {
            tf_call->resetDirty();

            m_gpu_materials->updateMaterialTexture(0, 0, tf_texture_name);
        }
    }

    mtl_call->setMaterialStorage(m_gpu_materials);

    // set update?

    return true;
}

} // namespace archvis
} // namespace megamol
