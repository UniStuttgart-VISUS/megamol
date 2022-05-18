/*
 * ModuleAutoDescription.h
 * Copyright (C) 2015 by Sebastian Grottel
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED
#pragma once

#include "mmcore/factories/ModuleAutoDescription.h"

namespace megamol {
namespace core {

#ifdef _MSC_VER
#ifndef STRINGIZE
#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#endif
#pragma message(__FILE__ "(" STRINGIZE(__LINE__) ") : warning: Both '#include \"mmcore/ModuleAutoDescription.h\"' and '::megamol::core::ModuleAutoDescription' are deprecated! Update to '#include \"mmcore/factories/ModuleAutoDescription.h\"' and '::megamol::core::factories::ModuleAutoDescription'.")
#else
#pragma GCC warning \
    "Both '#include \"mmcore/ModuleAutoDescription.h\"' and '::megamol::core::ModuleAutoDescription' are deprecated! Update to '#include \"mmcore/factories/ModuleAutoDescription.h\"' and '::megamol::core::factories::ModuleAutoDescription'."
#endif

/*
 * Some C++11 magic for the lazy asses
 */
template<class T>
using ModuleAutoDescription = ::megamol::core::factories::ModuleAutoDescription<T>;

} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED */
