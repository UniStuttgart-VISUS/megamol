#include "datRaw_half.h"

/* gcc 4.X bug, see datRaw_half.h */
#ifdef DATRAW_HALF_DO_NOT_INLINE
/*
	convert float to half float and vice versa
	based on code posted by gking on
    http://www.opengl.org/discussion_boards/ubb/Forum3/HTML/008786.html
*/

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
	fresult = *(float*)&result;
	return fresult;
}

DR_HALF floatToHalf(float val)
{
	const DR_INT f = *(DR_INT*)&val;
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
				datRaw_logError("logical error in denorm creation!\n");
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
#endif
