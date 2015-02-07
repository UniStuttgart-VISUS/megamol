/*
 * CallAutoDescription.h
 * Copyright (C) 2015 by Sebastian Grottel
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED
#pragma once

#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {

#ifdef _MSC_VER
#pragma message( "Both '#include \"mmcore/CallAutoDescription.h\"' and '::megamol::core::CallAutoDescription' are deprecated! Update to '#include \"mmcore/factories/CallAutoDescription.h\"' and '::megamol::core::factories::CallAutoDescription'." )
#else
#pragma GCC warning "Both '#include \"mmcore/CallAutoDescription.h\"' and '::megamol::core::CallAutoDescription' are deprecated! Update to '#include \"mmcore/factories/CallAutoDescription.h\"' and '::megamol::core::factories::CallAutoDescription'."
#endif

    /*
     * Some C++11 magic for the lazy asses
     */
    template <class T>
    using CallAutoDescription = ::megamol::core::factories::CallAutoDescription<T>;

}
}

#endif /* MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED */
