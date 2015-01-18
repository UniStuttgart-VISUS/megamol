/*
 * testfloat16.cpp  04.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testfloat16.h"
#include "testhelper.h"

#include <cfloat>
#include <cmath>
#include <limits>

#include "vislib/Float16.h"


#define DR_HALF UINT16
#define DR_INT INT32
#define DR_UINT UINT32

float halfToFloat(DR_HALF val)
{
	const DR_INT h = 0x0 | (DR_UINT)val;
	const DR_INT h_m = h & 0x3ff;
	const DR_INT h_e = (h >> 10) & 0x1f;
	const DR_INT h_s = (h >> 15) & 0x1;
	DR_INT f_e, f_m;
	DR_UINT result;
	float fresult;

	/*  handle special cases */
	if ((h_e == 0) && (h_m == 0)) {
		/* zero */
		f_m = 0;
		f_e = 0;
	} else if (h_e == 0 && h_m != 0) {
		/* denorm -- denorm half will fit in non-denorm float */
		const float half_denorm = (1.0f / 16384.0f); /* 2^-14 */
		float mantissa = ((float)h_m) / 1024.0f;
		float sgn = h_s ? -1.0f : 1.0f;
		return sgn*mantissa*half_denorm;
	} else if ((h_e == 31) && (h_m == 0)) {
		/* infinity */
		f_e = 0xff;
		f_m = 0x0;
	} else if ((h_e == 31) && (h_m != 0)) {
		/* NaN */
		f_e = 0xff;
		f_m = 0x1;
	} else {
		f_e = h_e + 112;
		f_m = (h_m << 13);
	}
	result = (h_s << 31) | (f_e << 23) | f_m;
	
	/*fprintf(stderr, "halfToFloat: 0x%x -> %f\n", val,
	 * *((float*)(void*)&result));*/
	 fresult = *((float*)(void*)&result);
	return fresult;
}

DR_HALF floatToHalf(float val)
{
	const DR_INT f = *(DR_INT*)(void*)&val;
	const DR_INT f_m = f & 0x7fffff;
	const DR_INT f_e = (f >> 23) & 0xff;
	const DR_INT f_s = (f >> 31) & 0x1;
	DR_INT h_e = 0, h_m = 0;

	/* handle special cases */
	if ((f_e == 0) && (f_m == 0)) {
		/* zero */
		/* h_m = 0; already set */
		/* h_e = 0; already set */
	} else if ((f_e == 0) && (f_m != 0)) {
		/* denorm -- denorm float maps to 0 half h_m = 0 */
		/* already set h_e = 0; */
	} else if ((f_e == 0xff) && (f_m == 0)) {
		/* infinity */
		/* h_m = 0; already set */
		h_e = 31;
	} else if ((f_e == 0xff) && (f_m != 0)) {
		/* NaN */
		h_m = 1;
		h_e = 31;
	} else {
		/* regular number */
		DR_INT new_exp = f_e - 127;
		if (new_exp < -24) {
			/* this maps to 0 */
			/* h_m = 0; already set */
			/* h_e = 0; already set */
		}

		if (new_exp < -14) {
			/* this maps to a denorm */
			/* h_e = 0; already set */
			DR_UINT exp_val = (DR_UINT) (-14 - new_exp); /* 2^-exp_val */
			switch (exp_val) {
			case 0:
//				datRaw_logError("logical error in denorm creation!\n");
				/* h_m = 0; already set */
				break;
			case 1:
				h_m = 512 + (f_m >> 14);
				break;
			case 2:
				h_m = 256 + (f_m >> 15);
				break;
			case 3:
				h_m = 128 + (f_m >> 16);
				break;
			case 4:
				h_m = 64 + (f_m >> 17);
				break;
			case 5:
				h_m = 32 + (f_m >> 18);
				break;
			case 6:
				h_m = 16 + (f_m >> 19);
				break;
			case 7:
				h_m = 8 + (f_m >> 20);
				break;
			case 8:
				h_m = 4 + (f_m >> 21);
				break;
			case 9:
				h_m = 2 + (f_m >> 22);
				break;
			case 10:
				h_m = 1;
				break;
			}
		} else if (new_exp > 15) {
			/* map this value to infinity */
			/* h_m = 0; already set */
			h_e = 31;
		} else {
			h_e = new_exp + 15;
			h_m = (f_m >> 13);
		}
	}

	return (f_s << 15) | (h_e << 10) | h_m;
	
}


void TestFloat16(void) {
    using vislib::math::Float16;
    
    float fltIn = 0.0f;
    float fltOut = 0.0f;
    float fltOutThomas = 0.0f;
    UINT16 hlf = 0;
    UINT16 hlfThomas;

    fltIn = 0.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert 0.0f to half", hlf, UINT16(0));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 0.0f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back 0.0h to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = 1.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert 1.0f to half", hlf, UINT16(15 << 10));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 1.0f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back 1.0h to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = -1.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert -1.0f to half", hlf, UINT16((1 << 15) | (15 << 10)));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for -1.0f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back -1.0h to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = 1.0f / 2.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert 1/2f to half", hlf, UINT16(7 << 11));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 1/2f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back 1/2h to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = 3.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert 3.0f to half", hlf, UINT16(16896));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 3.0f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back 3.0f to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = 65504.0f;
    hlf = Float16::FromFloat32(fltIn);
    ::AssertEqual("Convert 65504.0f to half", hlf, UINT16(31743));
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 65504.0f", hlf, hlfThomas);

    fltOut = Float16::ToFloat32(hlf);
    ::AssertEqual("Convert back 65504.0f to float", fltIn, fltOut);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = 1.0f / 3.0f;
    hlf = Float16::FromFloat32(fltIn);
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 1/3f", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);

    
    fltIn = 1.0f / 10000.0f;
    hlf = Float16::FromFloat32(fltIn);
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 1/10000f", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);

    fltIn = 1.0f / 100000.0f;
    hlf = Float16::FromFloat32(fltIn);
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 1/100000f", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = float(Float16::MAX);
    hlf = Float16::FromFloat32(fltIn);
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for Float16::MAX", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = float(Float16::MIN);
    hlf = Float16::FromFloat32(fltIn);
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for Float16::MIN", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);


    fltIn = float(2.0 * Float16::MAX + 10.0f);
    hlf = Float16::FromFloat32(fltIn);
    ::AssertTrue("Float16::MAX + 10 is infinity", Float16(2.0 * Float16::MAX).IsInfinity());
    hlfThomas = ::floatToHalf(fltIn);
    ::AssertEqual("Regression test with Thomas' impl for 2 * Float16::MAX", hlf, hlfThomas);
    fltOut = Float16::ToFloat32(hlf);
    fltOutThomas = ::halfToFloat(hlfThomas);
    ::AssertEqual("Regression test for convert back", fltOut, fltOutThomas);

    ::AssertTrue("float16 of ::sqrtf(-1.0f) is not a number", Float16(::sqrtf(-1.0f)).IsNaN());
    ::AssertTrue("float16 of std::numeric_limits<float>::quiet_NaN() is not a number", Float16(std::numeric_limits<float>::quiet_NaN()).IsNaN());
}

#undef DR_HALF
#undef DR_INT
#undef DR_UINT
