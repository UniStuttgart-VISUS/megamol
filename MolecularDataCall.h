/*
 * MolecularDataCall.h
 *
 * Copyright (C) 2015 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore\moldyn\MolecularDataCall.h"

namespace megamol {
namespace protein {

#ifdef _MSC_VER
#pragma message( "Both '#include \"MolecularDataCall.h\"' and '::megamol::protein::MolecularDataCall' are deprecated! Update to '#include \"mmcore/moldyn/MolecularDataCall.h\"' and '::megamol::core::moldyn::MolecularDataCall'." )
#else
#pragma GCC warning "Both '#include \"MolecularDataCall.h\"' and '::megamol::protein::MolecularDataCall' are deprecated! Update to '#include \"mmcore/moldyn/MolecularDataCall.h\"' and '::megamol::core::moldyn::MolecularDataCall'."
#endif

    /*
    * Some C++11 magic for the lazy asses
    */
    typedef ::megamol::core::moldyn::MolecularDataCall MolecularDataCall;
    typedef ::megamol::core::moldyn::MolecularDataCallDescription MolecularDataCallDescription;

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED */
