/*
 * TunnelResidueDataCall.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "protein_calls/TunnelResidueDataCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;

/*
 * TunnelResidueDataCall::TunnelResidueDataCall
 */
TunnelResidueDataCall::TunnelResidueDataCall(void) : AbstractGetData3DCall(), tunnels(nullptr), numTunnels(0) {}

/*
 * TunnelResidueDataCall::TunnelResidueDataCall
 */
TunnelResidueDataCall::~TunnelResidueDataCall(void) {
    // intentionally empty
}
