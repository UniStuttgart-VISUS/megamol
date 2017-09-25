/*
 * CoreHandle.cpp
 *
 * Copyright (C) 2006 - 2016 MegaMol Team.
 * All rights reserved
 */

#include "stdafx.h"
#include "CoreHandle.h"
#include "mmcore/api/MegaMolCore.h"
#include <cstring>
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include <memory>


/*
 * megamol::console::CoreHandle::MegaMolHandle
 */
megamol::console::CoreHandle::CoreHandle(void) 
        : hndl(nullptr), size(0) {
    // intentionally empty
}

megamol::console::CoreHandle::CoreHandle(CoreHandle&& src)
        : hndl(src.hndl), size(src.size) {
    src.hndl = nullptr;
    src.size = 0;
}

/*
 * megamol::console::CoreHandle::~MegaMolHandle
 */
megamol::console::CoreHandle::~CoreHandle(void) {
    this->DestroyHandle();
    ARY_SAFE_DELETE(this->hndl);
    this->size = 0;
}


/*
 * megamol::console::CoreHandle::DestroyHandle
 */
void megamol::console::CoreHandle::DestroyHandle(void) {
    ::mmcDisposeHandle(this->hndl); // this is ugly, but it works, since it is
    ::memset(this->hndl, 0, this->size);
}


/*
 * megamol::console::CoreHandle::operator void *
 */
megamol::console::CoreHandle::operator void *(void) const { 
    if (this->hndl == nullptr) {
        unsigned int size = ::mmcGetHandleSize();
        if (this->size < size) {
            ASSERT(!this->IsValid());

            this->size = size;

            delete[] this->hndl;
            this->hndl = new unsigned char[this->size];
            ::memset(this->hndl, 0, this->size);
        }
    }
    return static_cast<void*>(this->hndl);
}

megamol::console::CoreHandle& megamol::console::CoreHandle::operator=(CoreHandle&& src) {
    if (&src == this) return *this;
    if (hndl != nullptr) {
        DestroyHandle();
        delete[] hndl;
    }
    hndl = src.hndl;
    size = src.size;
    src.hndl = nullptr;
    src.size = 0;
    return *this;
}
