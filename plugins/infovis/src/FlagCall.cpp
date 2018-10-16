#include "stdafx.h"
#include "infovis/FlagCall.h"

using namespace megamol;
using namespace megamol::infovis;

/*
 * FlagCall::CallForGetFlags
 */
const unsigned int FlagCall::CallForGetFlags = 0;


/*
 * FlagCall::CallForSetFlags
 */
const unsigned int FlagCall::CallForSetFlags = 1;


/*
 * FlagCall::CallForResetDirty
 */
const unsigned int FlagCall::CallForResetDirty = 2;

/*
 *	FlagCall:FlagCall
 */
FlagCall::FlagCall(void) : flags(NULL), isDirty(false) {}

/*
 *	FlagCall::~FlagCall
 */
FlagCall::~FlagCall(void) { flags = nullptr; }
