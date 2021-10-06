/*
 * PNGDataSource.cpp
 *
 * Copyright (C) 2021 Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "PNGDataSource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FilePathParam.h"

#include "compositing/CompositingCalls.h"

#include "png.h"

using namespace megamol;
using namespace megamol::compositing;

PNGDataSource::PNGDataSource(void) : core::Module()
    , m_filename_mlot("filename", "Filename to read from")
    , m_output_tex_slot("getData", "Slot providing the data")
    , m_version(0)
{
    this->m_filename_mlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_mlot);

    this->m_output_tex_slot.SetCallback(
       CallTexture2D::ClassName(), "GetData", &PNGDataSource::getDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);
}

PNGDataSource::~PNGDataSource(void) {
    this->Release();
}

bool PNGDataSource::create(void) {
    // nothing to do
    return true;
}

void PNGDataSource::release(void) {
}

bool PNGDataSource::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);

    png_

    ++m_version;

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}
