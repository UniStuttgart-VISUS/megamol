/*
 * CallbackSlot.cpp
 *
 * Copyright (C) 2008-2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallbackSlot.h"


/*
 * megamol::wgl::CallbackSlot::CallbackSlot
 */
megamol::wgl::CallbackSlot::CallbackSlot(void) : one(NULL), other(NULL) {
}


/*
 * megamol::wgl::CallbackSlot::~CallbackSlot
 */
megamol::wgl::CallbackSlot::~CallbackSlot(void) {
    this->one = NULL;
    if (this->other != NULL) {
        SAFE_DELETE(this->other);
    }
}


/*
 * megamol::wgl::CallbackSlot::Call
 */
void megamol::wgl::CallbackSlot::Call(megamol::wgl::ApiHandle& caller,
        void *parameters) {
    if (this->one == NULL) return;
    this->one(caller.UserData, parameters);
    if (this->other == NULL) return;
    SIZE_T cnt = this->other->Count();
    for (SIZE_T i = 0; i < cnt; i++) {
        (*this->other)[i](caller.UserData, parameters);
    }
}


/*
 * megamol::wgl::CallbackSlot::Clear
 */
void megamol::wgl::CallbackSlot::Clear(void) {
    this->one = NULL;
    SAFE_DELETE(this->other);
}


/*
 * megamol::wgl::CallbackSlot::Register
 */
void megamol::wgl::CallbackSlot::Register(mmvCallback function) {
    if (this->one == NULL) {
       this->one = function;
    } else {
        if (this->other == NULL) {
            this->other = new vislib::Array<mmvCallback>();
            this->other->Append(function);
        }
    }
}


/*
 * megamol::wgl::CallbackSlot::Unregister
 */
void megamol::wgl::CallbackSlot::Unregister(mmvCallback function) {
    if (this->one == function) {
        if (this->other == NULL) {
            this->one = NULL;
        } else {
            this->one = this->other->First();
            this->other->RemoveFirst();
            if (this->other->Count() == 0) {
                SAFE_DELETE(this->other);
            }
        }
    } else if (this->other != NULL) {
        INT_PTR idx = this->other->IndexOf(function);
        if (idx != vislib::Array<mmvCallback>::INVALID_POS) {
            this->other->RemoveAt(idx);
        }
    }
}
