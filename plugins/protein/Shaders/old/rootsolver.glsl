// global constants
float doubtol = 0.00000001;		// min
float doubmin = 0.0;					// zero
float doubmax = 1000000000.0;	// max (inf)

// function prototypes
float acos3( float x);
float cubic( float p, float q, float r);
float curoot( float x);
void computeConstants();
int ferrari( float a, float b, float c, float d, out vec4 rts);
int neumark( float a, float b, float c, float d, out vec4 rts);
int quadratic( float b, float c, out vec4 rts, float dis);


/**
 *	Solve quartic equation using either quadratic, Ferrari's or Neumark's algorithm.
 *
 *	input:
 *		a, b, c, d - coefficients of equation.
 *	output:
 *		[return nquar] - number of real roots.
 *		rts - array of root values.
 *
 *	calls:
 *		quadratic, ferrari, neumark.
 */
int quartic( float a, float b, float c, float d, out vec4 rts)
{
	int j, k, nq, nr;
	float odd, even;
	vec4 roots;

	if( a < 0.0 )
		odd = -a;
	else
		odd = a;
	if( c < 0.0 )
		odd -= c;
	else
		odd += c;
	if( b < 0.0 )
		even = -b;
	else
		even = b;
	if( d < 0.0 )
		even -= d;
	else
		even += d;
	
	if( odd < even*doubtol )
	{
		nq = quadratic( b, d, roots, b*b-4.0*d);
		j = 0;
		if( nq == 1 )
		{
			if( roots.x > 0.0 )
			{
				rts.x = sqrt( roots.x);
				rts.y = -rts.x;
				++j; ++j;
			}
		}
		else if( nq == 2 )
		{
			if( roots.x > 0.0 )
			{
				rts.x = sqrt( roots.x);
				rts.y = -rts.x;
				++j; ++j;
			}
			if( roots.y > 0.0 )
			{
				rts.z = sqrt( roots.y);
				rts.w = -rts.z;
				++j; ++j;
			}
		}
		nr = j;
	}
	else
	{
		if( a < 0.0 )
			k = 1;
		else
			k = 0;
		if( b < 0.0 )
			k += k+1;
		else
			k +=k; 
		if( c < 0.0 )
			k += k+1;
		else
			k +=k; 
		if( d < 0.0 )
			k += k+1;
		else
			k +=k; 
		
		if( k == 0 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 1 ) nr = neumark( a, b, c, d, rts);
		else if( k == 2 ) nr = neumark( a, b, c, d, rts);
		else if( k == 3 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 4 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 5 ) nr = neumark( a, b, c, d, rts);
		else if( k == 6 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 7 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 8 ) nr = neumark( a, b, c, d, rts);
		else if( k == 9 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 10 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 11 ) nr = neumark( a, b, c, d, rts);
		else if( k == 12 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 13 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 14 ) nr = ferrari( a, b, c, d, rts);
		else if( k == 15 ) nr = ferrari( a, b, c, d, rts);
	}
	return nr;
} // ----- quartic -----

/**
 *	compute constant values
 */
void computeConstants()
{
	int j;
	doubtol = 1.0;
	for( j = 1; 1.0+doubtol > 1.0; ++j )
	{
		doubtol *= 0.5;
	}
	doubtol = sqrt( doubtol);
	
	doubmin = 0.5;
	for( j = 1; j <= 100; ++j )
	{
		doubmin = doubmin*doubmin;
		if( (doubmin*doubmin) <= (doubmin*doubmin*0.5) )
			break;
	}
	doubmax = 0.7/sqrt( doubmin);
} // ----- setconstants -----

/**
 *	solve the quartic equation
 *		x**4 + a*x**3 + b*x**2 + c*x + d = 0
 *
 *	input:
 *		a, b, c, d - coefficients of equation.
 *	output:
 *		[return nquar] - number of real roots.
 *		rts - array of root values.
 *
 *	calls:
 *		cubic
 *		quadratic
 */
int ferrari( float a, float b, float c, float d, out vec4 rts)
{
	rts = vec4( 0.0, 0.0, 0.0, 0.0);
	
	int nquar, n1, n2;
	float asq, ainv2;
	vec4 v1, v2;
	float p, q, r;
	float y;
	float e, f, esq, fsq, ef;
	float g, gg, h, hh;

	asq = a*a;

	p = b;
	q = a * c - 4.0 * d;
	r = ( asq - 4.0 * b) * d + c*c;
	y = cubic( p, q, r);

	esq = 0.25 * asq - b - y;
	if( esq < 0.0 )
		return(0);
	else
	{
		fsq = 0.25*y*y - d;
		if( fsq < 0.0 )
			return 0;
		else
		{
			ef = -(0.25*a*y + 0.5*c);
			if( ((a > 0.0) && (y > 0.0) && (c > 0.0))
				|| ((a > 0.0) && (y < 0.0) && (c < 0.0))
				|| ((a < 0.0) && (y > 0.0) && (c < 0.0))
				|| ((a < 0.0) && (y < 0.0) && (c > 0.0))
				||  (a == 0.0) || (y == 0.0) || (c == 0.0) )
			// use ef
			{
				if( (b < 0.0) && (y < 0.0) && (esq > 0.0) )
				{
					e = sqrt( esq);
					f = ef/e;
				}
				else if( (d < 0.0) && (fsq > 0.0) )
				{
					f = sqrt( fsq);
					e = ef/f;
				}
				else
				{
					e = sqrt( esq);
					f = sqrt( fsq);
					if( ef < 0.0 ) f = -f;
				}
			}
			else
			{
				e = sqrt( esq);
				f = sqrt( fsq);
				if( ef < 0.0 ) f = -f;
			}
			// note that e >= 0.0
			ainv2 = a*0.5;
			g = ainv2 - e;
			gg = ainv2 + e;
			if( ((b > 0.0) && (y > 0.0))
				|| ((b < 0.0) && (y < 0.0)) )
			{
				if( ( a > 0.0) && (e != 0.0) )
					g = (b + y)/gg;
				else if( e != 0.0 )
					gg = (b + y)/g;
			}
			if( (y == 0.0) && (f == 0.0) )
			{
				h = 0.0;
				hh = 0.0;
			}
			else if( ((f > 0.0) && (y < 0.0))
				|| ((f < 0.0) && (y > 0.0)) )
			{
				hh = -0.5*y + f;
				h = d/hh;
			}
			else
			{
				h = -0.5*y - f;
				hh = d/h;
			}
			n1 = quadratic( gg, hh, v1, gg*gg - 4.0 * hh);
			n2 = quadratic( g, h, v2, g*g - 4.0 * h);
			nquar = n1 + n2;
			rts.x = v1.x;
			rts.y = v1.y;
			if( n1 == 0 )
			{
				rts.x = v2.x;
				rts.y = v2.y;
			}
			else
			{
				rts.z = v2.x;
				rts.w = v2.y;
			}
			return nquar;
		}
	}
} // ----- ferrari -----


/**
 *	solve the quartic equation
 *		x**4 + a*x**3 + b*x**2 + c*x + d = 0
 *
 *	input:
 *		a, b, c, e - coefficients of equation.
 *
 *	output:
 *		[return nquar] - number of real roots.
 *		rts - array of root values.
 *
 * 	calls:
 *		cubic
 *		quadratic
 */
int neumark( float a, float b, float c, float d, out vec4 rts)
{
	int nquar, n1, n2;
	float y, g, gg, h, hh, gdis, gdisrt, hdis, hdisrt, g1, g2, h1, h2;
	float bmy, gerr, herr, y4, d4, bmysq;
	vec4 v1, v2;
	float asq;
	float p,q,r;
	float hmax,gmax;

	asq = a*a ;

	p = -b*2.0;
	q = b*b + a*c - 4.0*d;
	r = (c - a*b)*c + asq*d;
	y = cubic( p, q, r);

	bmy = b - y;
	y4 = y*4.0;
	d4 = d*4.0;
	bmysq = bmy*bmy ;
	gdis = asq - y4 ;
	hdis = bmysq - d4 ;
	if( (gdis < 0.0) || (hdis < 0.0) )
		return 0;
	else
	{
		g1 = a*0.5;
		h1 = bmy*0.5;
		gerr = asq + y4;
		herr = hdis;
		if( d > 0.0 )
			herr = bmysq + d4;
		if( (y < 0.0) || (herr*gdis > gerr*hdis) )
		{
			gdisrt = sqrt(gdis);
			g2 = gdisrt*0.5;
			if( gdisrt != 0.0 )
				h2 = (a*h1 - c)/gdisrt;
			else
				h2 = 0.0;
		}
		else
		{
			hdisrt = sqrt(hdis);
			h2 = hdisrt*0.5;
			if( hdisrt != 0.0 )
				g2 = (a*h1 - c)/hdisrt;
			else
				g2 = 0.0;
		} 
		//note that in the following, the tests ensure non-zero denominators
		h = h1 - h2 ;
		hh = h1 + h2 ;
		hmax = hh ;
		if( hmax < 0.0 ) hmax = -hmax;
		if( hmax < h ) hmax = h;
		if( hmax < -h ) hmax = -h;
		if( (h1 > 0.0 ) && (h2 > 0.0)) h = d/hh;
		if( (h1 < 0.0 ) && (h2 < 0.0)) h = d/hh;
		if( (h1 > 0.0 ) && (h2 < 0.0)) hh = d/h;
		if( (h1 < 0.0 ) && (h2 > 0.0)) hh = d/h;
		if( h > hmax ) h = hmax;
		if( h < -hmax ) h = -hmax;
		if( hh > hmax ) hh = hmax;
		if( hh < -hmax ) hh = -hmax;

		g = g1 - g2;
		gg = g1 + g2;
		gmax = gg;
		if( gmax < 0.0 ) gmax = -gmax;
		if( gmax < g ) gmax = g;
		if( gmax <  -g ) gmax = -g;
		if( (g1 > 0.0) && (g2 > 0.0) ) g = y/gg;
		if( (g1 < 0.0) && (g2 < 0.0) ) g = y/gg;
		if( (g1 > 0.0) && (g2 < 0.0) ) gg = y/g;
		if( (g1 < 0.0) && (g2 > 0.0) ) gg = y/g;
		if( g > gmax ) g = gmax;
		if( g <  -gmax ) g = -gmax;
		if( gg > gmax ) gg = gmax;
		if( gg <  -gmax ) gg = -gmax;

		n1 = quadratic( gg, hh, v1, gg*gg - 4.0*hh);
		n2 = quadratic( g, h, v2, g*g - 4.0*h);
		nquar = n1+n2;
		rts.x = v1.x;
		rts.x = v1.y;
		if( n1 == 0 )
		{
			rts.x = v2.x;
			rts.y = v2.y;
		}
		else
		{
			rts.z = v2.x;
			rts.w = v2.y;
		}

		return nquar;
	}
} // ----- neumark -----


/**
 *	solve the quadratic equation
 *		x**2+b*x+c = 0
 *
 *	input:
 *		b, c - coefficients of equation.
 *	output:
 *		[return nquad] - number of real roots.
 *		rts - array of root values.+
 *		
 *	called by:
 *		ferrari
 *		neumark
 */
int quadratic( float b, float c, out vec4 rts, float dis)
{
	int nquad;
	float rtdis;

	if( dis >= 0.0 )
	{
		nquad = 2;
		rtdis = sqrt( dis) ;
		if( b > 0.0 )
			rts.x = ( -b - rtdis) * 0.5;
		else
			rts.x = ( -b + rtdis) * 0.5;
		if( rts.x == 0.0 )
			rts.y = -b;
		else
			rts.y = c/rts.x;
	}
	else
	{
		nquad = 0;
		rts.x = 0.0;
		rts.y = 0.0;
	}
	return nquad;
} // ----- quadratic -----


/**
 *	find the lowest real root of the cubic equation
 *		x**3 + p*x**2 + q*x + r = 0 
 *
 *	input parameters:
 *		p, q, r - coefficients of cubic equation. 
 *	output:
 *		cubic - a real root.
 *
 *	calls:
 *		acos3
 *		curoot
 *	called by:
 *		ferrari
 *		neumark
 */
float cubic( float p, float q, float r)
{	
	int nrts;
	float po3, po3sq, qo3;
	float uo3, u2o3, uo3sq4, uo3cu4;
	float v, vsq, wsq;
	float m, mcube, n;
	float muo3, s, scube, t, cosk, sinsqk;
	float root;

	m = 0.0;
	nrts = 0;
	if( (p > doubmax) || (p <  -doubmax) )
		root = -p;
	else
	{
		if( (q > doubmax) || (q <  -doubmax) )
		{
			if (q > 0.0)
				root = -r/q;
			else
				root = -sqrt( -q);
		}
		else
		{
			if( (r > doubmax) || (r <  -doubmax) )
				root = -curoot( r);
			else
			{
				po3 = p * (1.0/3.0);
				po3sq = po3*po3 ;
				if( po3sq > doubmax )
					root = -p;
				else
				{
					v = r + po3*(po3sq + po3sq - q) ;
					if( (v > doubmax) || (v < -doubmax) )
						root = -p;
					else
					{
						vsq = v*v ;
						qo3 = q * (1.0/3.0);
						uo3 = qo3 - po3sq ;
						u2o3 = uo3 + uo3 ;
						if( (u2o3 > doubmax) || (u2o3 < -doubmax) )
						{
							if (p == 0.0)
							{
								if (q > 0.0)
									root = -r/q;
								else
									root = -sqrt( -q);
							}
							else
								root = -q/p;
						}
						uo3sq4 = u2o3 * u2o3 ;
						if( uo3sq4 > doubmax)
						{
							if (p == 0.0)
							{
								if( q > 0.0 )
									root = -r/q;
								else
									root = -sqrt( abs( q));
							}
							else
								root = -q/p;
						}
						uo3cu4 = uo3sq4 * uo3;
						wsq = uo3cu4 + vsq;
						if( wsq >= 0.0 )
						{
							// cubic has one real root
							nrts = 1;
							if( v <= 0.0 )
								mcube = ( -v + sqrt( wsq))*0.5;
							if( v  > 0.0 )
								mcube = ( -v - sqrt( wsq))*0.5;
							m = curoot( mcube);
							if( m != 0.0 )
								n = -uo3/m;
							else
								n = 0.0;
							root = m + n - po3;
						}
						else
						{
							nrts = 3;
							// cubic has three real roots
							if( uo3 < 0.0 )
							{
								muo3 = -uo3;
								s = sqrt( muo3);
								scube = s*muo3;
								t =  -v/(scube+scube);
								cosk = acos3( t);
								if( po3 < 0.0 )
									root = (s+s)*cosk - po3;
								else
								{
									sinsqk = 1.0 - cosk*cosk;
									if( sinsqk < 0.0 )
										sinsqk = 0.0;
									root = s*( -cosk - sqrt( 3.0)*sqrt( sinsqk)) - po3;
								}
							}
							else
								// cubic has multiple root
								root = curoot( v) - po3;
						}
					}
				}
			}
		}
	}
	return root;
} // ----- cubic -----


/** 
 *	find cube root of x.
 *
 *	called by:
 *		cubic 
 */
float curoot( float x)
{
	float value;
	float absx;
	int neg;

	neg = 0;
	absx = x;
	if( x < 0.0 )
	{
		absx = -x;
		neg = 1;
	}
	value = exp( log( absx)*(1.0/3.0));
	if( neg == 1 )
		value = -value;
	return value;
} // ----- curoot -----


/** 
 * find cos(acos(x)/3) 
 *
 * called by:
 *	cubic 
*/
float acos3( float x)
{
	return cos( acos( x)*(1.0/3.0));
} // ----- acos3 -----


/**
 *	solve the quartic equation
 *		x**4 + a*x**3 + b*x**2 + c*x + d = 0
 *
 *	input:
 *		a, b, c, d - coefficients of equation.
 *	output:
 *		[return nquar] - number of real roots.
 *		rts - array of root values.
 *
 *	calls:
 *		cubic
 *		quadratic
 */
int simpleFerrari( float a, float b, float c, float d, out vec4 rts)
{
	int nquar, n1, n2;
	float asq, y;
	vec4 v1, v2;
	float p, q, r;
	float e, f, esq, fsq;
	float g, gg, h, hh;

	asq = a*a;

	p = -b;
	q = a*c-4.0*d;
	r = -asq*d - c*c + 4.0*b*d;
	y = cubic( p, q, r);

	esq = 0.25*asq - b + y;
	fsq = 0.25*y*y - d;
	if( esq < 0.0 )
		return 0;
	else
	{
		if( fsq < 0.0 )
			return 0;
		else
		{
			e = sqrt( esq);
			f = sqrt( fsq);
			g = 0.5*a - e;
			h = 0.5*y - f;
			gg = 0.5*a + e;
			hh = 0.5*y + f;
			n1 = quadratic( gg, hh, v1, gg*gg - 4.0*hh) ;
			n2 = quadratic( g, h, v2, g*g - 4.0*h) ;
			nquar = n1 + n2;
			rts.x = v1.x;
			rts.y = v1.y;
			if( n1 == 0 )
			{
				rts.x = v2.x;
				rts.y = v2.y;
			}
			else
			{
				rts.z = v2.x;
				rts.w = v2.y;
			}
			return nquar;
		}
	}
} // ----- simple -----


/**
 *	solve the quartic equation
 *		x**4 + a*x**3 + b*x**2 + c*x + d = 0
 *
 *	input:
 *		a, b, c, d - coefficients of equation.
 *	output:
 *		[return nquar] - number of real roots.
 *		rts - array of root values.
 *
 *	calls:
 *		cubic
 *		quadratic
 */
int descartes( float a, float b, float c, float d, out vec4 rts)
{
	int nrts;
	int r1,r2;
	vec4 v1, v2;
	float y;
	float p,q,r;
	float A,B,C;
	float m,n1,n2;
	float d3o8,d3o256;
	float inv8,inv16;
	float asq;
	float Binvm;

	d3o8 = 3.0/8.0;
	inv8 = 1.0/8.0;
	inv16 = 1.0/16.0;
	d3o256 = 3.0/256.0;

	asq = a*a;

	A = b - asq*d3o8;
	B = c + a*(asq*inv8 - b*0.5);
	C = d + asq*(b*inv16 - asq*d3o256) - a*c*0.25;

	p = 2.0*A;
	q = A*A - 4.0*C;
	r = -B*B;

	y = cubic( p, q, r) ;
	if( y <= 0.0 ) 
		nrts = 0;
	else
	{
		m = sqrt( y);
		Binvm = B/m;
		n1 = ( y + A + Binvm)*0.5;
		n2 = ( y + A - Binvm)*0.5;
		r1 = quadratic(-m, n1, v1, y-4.0*n1);
		r2 = quadratic( m, n2, v2, y-4.0*n2);
		rts.x = v1.x-a*0.25;
		rts.y = v1.y-a*0.25;
		if( r1 == 0 )
		{
			rts.x = v2.x-a*0.25;
			rts.y = v2.y-a*0.25;
		}
		else
		{
			rts.z = v2.x-a*0.25;
			rts.w = v2.y-a*0.25;
		}
		nrts = r1+r2;
	} 
	return nrts;
} // ----- descartes -----
