/*
 * TunnelResidueDataCall.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TunnelResidueDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;

/*
 * TunnelResidueDataCall::TunnelResidueDataCall
 */
TunnelResidueDataCall::TunnelResidueDataCall(void) : AbstractGetData3DCall(), tunnels(nullptr), numTunnels(0) {
}

/*
 * TunnelResidueDataCall::TunnelResidueDataCall
 */
TunnelResidueDataCall::~TunnelResidueDataCall(void) {
	// intentionally empty
}