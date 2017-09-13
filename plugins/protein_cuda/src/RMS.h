/** _______________________________________________________________________
 *
 *                        RDPARM/PTRAJ: 2008
 *  _______________________________________________________________________
 *
 *  This file is part of rdparm/ptraj.
 *
 *  rdparm/ptraj is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  rdparm/ptraj is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You can receive a copy of the GNU General Public License from
 *  http://www.gnu.org or by writing to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *  ________________________________________________________________________
 *
 *  CVS tracking:
 *
 *  $Header: /storage/disk2/case/cvsroot/amber11/src/ptraj/rms.h,v 10.0 2008/04/15 23:24:11 case Exp $
 *
 *  Revision: $Revision: 10.0 $
 *  Date: $Date: 2008/04/15 23:24:11 $
 *  Last checked in by $Author: case $
 *  ________________________________________________________________________
 *
 *
 *  CONTACT INFO: To learn who the code developers are, who to contact for
 *  more information, and to know what version of the code this is, refer
 *  to the CVS information and the include files (contributors.h && version.h)
 *
 *
 *
 *  ________________________________________________________________________
 *
 *  code modified for usage in megamol 0.3 (date: 2009/03/03)
 * 
 */

#ifndef RMSD_H_INCLUDED
#define RMSD_H_INCLUDED

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace protein_cuda {

#ifndef RMS_ROTATE_JACOBI3
#define ROTATE_JACOBI3(ARR,MAJ1,MIN1,MAJ2,MIN2) { \
            g = ARR[MAJ1 + MIN1]; \
            h = ARR[MAJ2 + MIN2]; \
            ARR[MAJ1 + MIN1] = g - s*(h+g*tau); \
            ARR[MAJ2 + MIN2] = h + s*(g-h*tau); }
#endif /* ROTATE_JACOBI3 */

/**
 * Normalize a 3D vector
 * 
 * @param a Vector which is normalized
 */
void Normalize(double a[3]);

/**
 * ?
 * 
 * @param mat        ...
 * @param Emat       ...
 * @param Evec       ...
 * @param Eigenvalue ...
 * 
 * @return ...
 */
int DiagEsort(double *mat, double *Emat, double *Evec[], double *Eigenvalue);

/**
 * Get jacobian of 3x3 matrix
 * 
 * @param a    ...
 * @param d    ...
 * @param v    ...
 * @param nrot ...
 *

 * @return ...
 */
int Jacobi3(double *a, double *d, double *v, int *nrot);

/**
* Fit (superimpose) position vectors (if necessary) with Kabsch algorithm
* (http://journals.iucr.org/a/issues/1976/05/00/a12999/a12999.pdf)
* and calculate RMS value for two given position vectors like in
* http://en.wikipedia.org/wiki/Root_mean_square_deviation_(bioinformatics.
* 
* Changes to original code in ptraj of AmberTools(10.0):
* - changed meaning of 'mode' and 'fit' (more intuitivly)
* - input data are now only two vectors containing x, y, and z values
* - rotation and translation vectors are only needed if mode == 1 or mode == 2
*
* if fit == 0, dispersion is calculated, no fit will be done.
* if fit == 1 and mode == 0, rms deviation is calculated but no structure will move.
* if fit == 1 and mode == 1, rms deviation is calculated, Vec moves back, but toFitVec's centroid moved to (0,0,0),
*                         as original functionality. Alignment will be done in the calling function.
* if fit == 1 and mode == 2, rms deviation is calculated and toFitVec will align to Vec. 
*
* @param n           Number of Positions (xyz) of each Vector.
* @param fit         True if position vectors should be fit.
* @param mode        See above ...
* @param mass        n weights for positions.
* @param mask        Vector with n element which describes if a position should be considered ('1') or not ('0').
* @param translation Matrix which stores the translation.
* @param rotation    Matrix which stores the rotation.
* @param Vec         Reference input Vector.
* @param toFitVec    Vector which is fit against Vec.
* 
* @return Return the calculated RMS value
*/

float CalculateRMS(unsigned int n, bool fit, unsigned int mode, float *mass, int *mask, 
                   float *toFitVec, float *Vec, float rotation[3][3], float translation[3]);


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* RMSD_H_INCLUDED */
