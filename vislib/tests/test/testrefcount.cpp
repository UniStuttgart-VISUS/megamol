/*
 * testrefcount.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#include "testrefcount.h"
#include "testhelper.h"

#include "vislib/SmartRef.h"



class Dowel : public vislib::ReferenceCounted {
public:
    inline Dowel() : vislib::ReferenceCounted() {}

protected:
    virtual ~Dowel();
};

Dowel::~Dowel() {}


void TestRefCount(void) {
    typedef vislib::SmartRef<Dowel> SmartDowel;

    SmartDowel d1(new Dowel(), false);
    AssertEqual("Increment reference count.", d1->AddRef(), UINT32(2));
    AssertEqual("Release referenced object.", d1->Release(), UINT32(1));

    {
        SmartDowel d2 = d1;
        AssertEqual("Increment reference count after creating new SmartRef.", d1->AddRef(), UINT32(3));
        AssertEqual("Release referenced object.", d1->Release(), UINT32(2));
    }

    AssertEqual("Increment reference count after deleting SmartRef.", d1->AddRef(), UINT32(2));
    AssertEqual("Release referenced object.", d1->Release(), UINT32(1));
}
