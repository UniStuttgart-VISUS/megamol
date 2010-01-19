/*
 * RenderNetMsg.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RenderNetMsg.h"

using namespace megamol::core;


/*
 * special::RenderNetMsg::RenderNetMsg
 */
special::RenderNetMsg::RenderNetMsg(void) : dat(RenderNetMsg::headerSize) {
    *this->dat.As<UINT32>() = 0;
    *this->dat.AsAt<UINT32>(4) = 0;
    *this->dat.AsAt<UINT64>(8) = 0;
}


/*
 * special::RenderNetMsg::RenderNetMsg
 */
special::RenderNetMsg::RenderNetMsg(UINT32 type, UINT32 id, SIZE_T size,
        const void *data) : dat(RenderNetMsg::headerSize + size) {
    *this->dat.As<UINT32>() = type;
    *this->dat.AsAt<UINT32>(4) = id;
    *this->dat.AsAt<UINT64>(8) = static_cast<UINT64>(size);
    if (data != NULL) {
        ::memcpy(this->dat.AsAt<void>(RenderNetMsg::headerSize), data, size);
    }
}


/*
 * special::RenderNetMsg::RenderNetMsg
 */
special::RenderNetMsg::RenderNetMsg(const special::RenderNetMsg& src)
        : dat(src.dat.GetSize()) {
    ::memcpy(this->dat.As<void>(), src.dat.As<void>(), src.dat.GetSize());
}


/*
 * special::RenderNetMsg::~RenderNetMsg
 */
special::RenderNetMsg::~RenderNetMsg(void) {
    this->dat.EnforceSize(0);
}


/*
 * special::RenderNetMsg::SetDataSize
 */
void special::RenderNetMsg::SetDataSize(SIZE_T size, bool keepData) {
    UINT32 type = *this->dat.As<UINT32>();
    UINT32 id = *this->dat.AsAt<UINT32>(4);
    this->dat.AssertSize(RenderNetMsg::headerSize + size, keepData);
    *this->dat.As<UINT32>() = type;
    *this->dat.AsAt<UINT32>(4) = id;
    *this->dat.AsAt<UINT64>(8) = static_cast<UINT64>(size);
}


/*
 * special::RenderNetMsg::operator=
 */
special::RenderNetMsg& special::RenderNetMsg::operator=(
        const special::RenderNetMsg& rhs) {
    if (this != &rhs) {
        this->dat.EnforceSize(rhs.dat.GetSize(), false);
        ::memcpy(this->dat.As<void>(), rhs.dat.As<void>(), rhs.dat.GetSize());
    }
    return *this;
}


