/*
 * testrefcount.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#include "testrefcount.h"
#include "testhelper.h"

#include "vislib/SmartRef.h"



class Dowel : public vislib::ReferenceCounted<vislib::SingleAllocator<Dowel> > {
public:
    inline Dowel() : vislib::ReferenceCounted<vislib::SingleAllocator<Dowel> >() {}

protected:
    virtual ~Dowel();
};

Dowel::~Dowel() {}


void TestRefCount(void) {
    typedef vislib::SmartRef<Dowel> SmartDowel;

    SmartDowel d1(new Dowel(), false);
}
