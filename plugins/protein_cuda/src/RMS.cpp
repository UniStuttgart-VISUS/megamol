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


#include "stdafx.h"
#include "RMS.h"
#include <cmath>
#include "vislib/sys/Log.h"

using namespace megamol;

/*
 *  protein_cuda::Normalize
 */
void protein_cuda::Normalize(double a[3])
{
    double b;

    b = 1.0/sqrt((double)(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]));
    a[0] *= b;
    a[1] *= b;
    a[2] *= b;
}


/*
 *  protein_cuda::DiagEsort
 */
int protein_cuda::DiagEsort(double *mat, double *Emat, double *Evec[], double *Eigenvalue)
{
    int njrot;
    int i, j, k, i3;
    double eigenvector[9], *eA, v;

    if(!Jacobi3(mat, Eigenvalue, eigenvector, &njrot)) 
    {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "RMS: DiagEsort - convergence failed! \n");
        return(0);
    }

    for(i = i3 = 0; i < 3; i++, i3 += 3)
        for (j=0; j<3; j++)
            Emat[i3+j] = eigenvector[j*3+i];

    for(i = 0; i < 3; i++)
        Evec[i] = (double *) &Emat[i*3];

    for(i = 0; i < 2; i++) 
    {
        v = Eigenvalue[k=i];
        for(j = i+1; j < 3; j++)
            if(Eigenvalue[j] > v)  
                v = Eigenvalue[k=j];

        if(k != i) 
        {
            Eigenvalue[k] = Eigenvalue[i];
            Eigenvalue[i] = v;
            eA = Evec[i];
            Evec[i] = Evec[k];
            Evec[k] = eA;
        }
    }
    return(1);
}


/*
 *  protein_cuda::Jacobi3
 */
int protein_cuda::Jacobi3(double *a, double *d, double *v, int *nrot)
{
    int  i, j, ip, iq, p3, j3;
    double  tresh, theta, tau, t, sm, s, h, g, c, b[3], z[3];

    for(ip = p3=0; ip < 3; ip++, p3 += 3) 
    {
        /*
         *  initialize the identity matrix 
         */
        for(iq = 0; iq < 3; iq++) 
            v[p3 + iq] = 0.0;

        v[p3 + ip] = 1.0;
        /* 
         *  initialize b and d to diagonal of a
         */
        b[ip] = d[ip] = a[p3 + ip];
        z[ip] = 0.0;
    }

    *nrot = 0;
    for(i = 0; i < 50; i++) 
    {
        sm = 0.0;
        for(ip = p3 = 0; ip < 2; ip++, p3 += 3) 
        {
            for(iq = ip+1; iq < 3; iq++)
                sm += fabs(a[p3 + iq]);
        }

        if(sm == 0.0) 
        {
            return(1);
        }
        if(i < 3) 
            tresh = sm * 0.2 / 9.0; /* on 1st three sweeps... */
        else       
            tresh = 0.0; /* thereafter... */

        for(ip = p3 = 0; ip < 2; ip++, p3 += 3) 
        {
            for(iq = ip+1; iq < 3; iq++) 
            {
                g = 100.0 * fabs(a[p3 + iq]);

                if((i > 3) && (fabs(d[ip])+g == fabs(d[ip])) && (fabs(d[iq])+g == fabs(d[iq]))) 
                {
                    a[p3 + iq] = 0.0;
                } 
                else if(fabs(a[p3 + iq]) > tresh) 
                {
                    h = d[iq]-d[ip];
                    if(fabs(h)+g == fabs(h))
                        t = a[p3 + iq] / h;
                    else 
                    {
                        theta = 0.5 * h / a[p3 + iq];
                        t = 1.0 / (fabs(theta)+(double)sqrt(1.0+theta*theta));
                        if (theta < 0.0) 
                            t = -t;
                    }
                    c = 1.0 / (double)sqrt(1.0 + t*t);
                    s = t * c;
                    tau = s / (1.0 + c);
                    h = t * a[p3 + iq];
                    z[ip] -= h;
                    z[iq] += h;
                    d[ip] -= h;
                    d[iq] += h;
                    a[p3 + iq] = 0.0;
                    for(j = j3 = 0; j <= ip-1; j++, j3 += 3) 
                        ROTATE_JACOBI3(a,j3,ip,j3,iq)
	                for(j = ip+1; j <= iq-1; j++) 
		                ROTATE_JACOBI3(a,p3,j,j*3,iq)
		            for(j = iq+1; j < 3; j++) 
		                ROTATE_JACOBI3(a,p3,j,iq*3,j)
		            for(j3 = 0; j3 < 9; j3 += 3) 
			            ROTATE_JACOBI3(v,j3,ip,j3,iq)

			        ++(*nrot);
                }
            }
        }
        for(ip = 0; ip < 3; ip++) 
        {
            b[ip] += z[ip];
            d[ip] = b[ip];
            z[ip] = 0.0;
        }
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "RMS: Jacobi3 - there are too many iterations! \n");
    return(0);
}


/*
 *  protein_cuda::CalculateRMS
 */
float protein_cuda::CalculateRMS(unsigned int n, bool fit, unsigned int mode, float *mass, int *mask, 
	               float *toFitVec, float *Vec, float rotation[3][3], float translation[3])
{
    int ierr=0;
    unsigned int i, j, k, modifiedCount;
    const char *err;
    double rms_return = 0.0; // do not know. however, better than uninitialized
    double *weights;
    double rot[9], rtr[9];
    int i3, k3;
    double mwss;
    double b[9], U[9];
    double *Evector[3], Eigenvalue[3], Emat[9];
    double x, y, z, xx, yy, zz;
    double total_mass;
    double sig3;
    double cp[3];
    double cofmX, cofmY, cofmZ;
    double cofmX1, cofmY1, cofmZ1;
    float xtemp, ytemp, ztemp;

    weights = new double[n];
    total_mass = 0.0;

    if(!fit) 
    {
        /*
         *  Don't do the fit, just calculate rmsd: don't calculate 
         *  any translation/rotation 
         */
        rms_return = 0.0;
        for(i = 0; i < n; i++) 
        {
            if (mask != NULL && mask[i] == 1) 
            {
                if (mass != NULL)
                    weights[i] = mass[i];
                else
                    weights[i] = 1.0;

                total_mass += weights[i];
                xx = Vec[3*i] - toFitVec[3*i];
                yy = Vec[3*i+1] - toFitVec[3*i+1];
                zz = Vec[3*i+2] - toFitVec[3*i+2];
                rms_return += weights[i]*(xx*xx + yy*yy + zz*zz);
            }
        }
        rms_return = sqrt(rms_return / total_mass);
        delete []weights;
        return (float) rms_return;
    }

    /*
     *  the rest below is for fit=1, i.e. calculate translation and
     *  rotation matrix as well as rmsd value of the fitted region 
     */

    for(i = 0, modifiedCount = n; i < n; i++) 
    {
        if((mask != NULL) && (mask[i] == 0)) 
        {
            weights[i] = 0.0;
            modifiedCount--;
        }
        else
        {
            if(mass != NULL)
                weights[i] = mass[i];
            else
                weights[i] = 1.0;

            total_mass += weights[i];
        }
    }

    if((mode == 1) || (mode == 2))
    {
        if((rotation == NULL) || (translation == NULL))
        {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "RMS: CalculateRMS - rotation matrix and translation vector are NULL ? \n");
        }
    }

    if(modifiedCount > 2) 
    {
        memset(rot, 0, sizeof(double) * 9);
        memset(rtr, 0, sizeof(double) * 9);
        memset(U,   0, sizeof(double) * 9);

        cofmX =  0.0;
        cofmY =  0.0;
        cofmZ =  0.0;
        cofmX1 = 0.0;
        cofmY1 = 0.0;
        cofmZ1 = 0.0;

        /*
         *  First shift the center of mass of all the atoms to be fit to
         *  the origin for both trajectory and reference coordinates.
         */
        for(k = 0; k < n; k++) 
        {
            cofmX += weights[k] * toFitVec[3*k];
            cofmY += weights[k] * toFitVec[3*k+1];
            cofmZ += weights[k] * toFitVec[3*k+2];
            cofmX1 += weights[k] * Vec[3*k];
            cofmY1 += weights[k] * Vec[3*k+1];
            cofmZ1 += weights[k] * Vec[3*k+2];
//            printf("RMS %f %f %f\n", Vec[3*k], Vec[3*k+1], Vec[3*k+2]);
        }

        cofmX /= total_mass;
        cofmY /= total_mass;
        cofmZ /= total_mass;
        cofmX1 /= total_mass;
        cofmY1 /= total_mass;
        cofmZ1 /= total_mass;

        for(k = 0; k < n; k++) 
        {
            toFitVec[3*k] -= (float)cofmX;
            toFitVec[3*k+1] -= (float)cofmY;
            toFitVec[3*k+2] -= (float)cofmZ;

            Vec[3*k] -= (float)cofmX1;
            Vec[3*k+1] -= (float)cofmY1;
            Vec[3*k+2] -= (float)cofmZ1;
        }

        mwss = 0.0;
        for (k = 0; k < n; k++) 
        {
            x  = toFitVec[3*k];
            y  = toFitVec[3*k+1];
            z  = toFitVec[3*k+2];
            xx = Vec[3*k];
            yy = Vec[3*k+1];
            zz = Vec[3*k+2];

            mwss += weights[k] * ( x*x + y*y + z*z + xx*xx + yy*yy + zz*zz );

            /*
            *  calculate the Kabsch matrix: R = (rij) = Sum(wn*yni*xnj) 
            */
            rot[0] += weights[k] * x * xx;
            rot[1] += weights[k] * x * yy;
            rot[2] += weights[k] * x * zz;

            rot[3] += weights[k] * y * xx;
            rot[4] += weights[k] * y * yy;
            rot[5] += weights[k] * y * zz;

            rot[6] += weights[k] * z * xx;
            rot[7] += weights[k] * z * yy;
            rot[8] += weights[k] * z * zz;
        }

        mwss *= 0.5f;   /* E0 = 0.5*Sum(wn*(xn^2+yn^2)) */

        /*
         *  calculate Kabsch multiplied by its transpose: RtR 
         */
        rtr[0] = rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2];
        rtr[1] = rot[0]*rot[3] + rot[1]*rot[4] + rot[2]*rot[5];
        rtr[2] = rot[0]*rot[6] + rot[1]*rot[7] + rot[2]*rot[8];
        rtr[3] = rot[3]*rot[0] + rot[4]*rot[1] + rot[5]*rot[2];
        rtr[4] = rot[3]*rot[3] + rot[4]*rot[4] + rot[5]*rot[5];
        rtr[5] = rot[3]*rot[6] + rot[4]*rot[7] + rot[5]*rot[8];
        rtr[6] = rot[6]*rot[0] + rot[7]*rot[1] + rot[8]*rot[2];
        rtr[7] = rot[6]*rot[3] + rot[7]*rot[4] + rot[8]*rot[5];
        rtr[8] = rot[6]*rot[6] + rot[7]*rot[7] + rot[8]*rot[8];

        if(!DiagEsort(rtr, Emat, Evector, Eigenvalue))
            return(0.0f);

        /*
         *  a3 = a1 x a2 
         */
        /*VOP_3D_COORDS_CROSS_PRODUCT(Evector[2][0], Evector[2][1], Evector[2][2], 
				    Evector[0][0], Evector[0][1], Evector[0][2],
				    Evector[1][0], Evector[1][1], Evector[1][2]);*/
        Evector[2][0] = (Evector[0][1] * Evector[1][2]) - (Evector[0][2] * Evector[1][1]);
        Evector[2][1] = (Evector[0][2] * Evector[1][0]) - (Evector[0][0] * Evector[1][2]);
        Evector[2][2] = (Evector[0][0] * Evector[1][1]) - (Evector[0][1] * Evector[1][0]);

        /*
         *  Evector dot transpose rot:  b = R.ak 
         */
        b[0] = Evector[0][0] * rot[0] + 
        Evector[0][1] * rot[3] + 
        Evector[0][2] * rot[6];
        b[1] = Evector[0][0] * rot[1] + 
        Evector[0][1] * rot[4] + 
        Evector[0][2] * rot[7];
        b[2] = Evector[0][0] * rot[2] + 
        Evector[0][1] * rot[5] + 
        Evector[0][2] * rot[8];

        Normalize(&b[0]);

        b[3] = Evector[1][0] * rot[0] + 
        Evector[1][1] * rot[3] + 
        Evector[1][2] * rot[6];
        b[4] = Evector[1][0] * rot[1] + 
        Evector[1][1] * rot[4] + 
        Evector[1][2] * rot[7];
        b[5] = Evector[1][0] * rot[2] + 
        Evector[1][1] * rot[5] + 
        Evector[1][2] * rot[8];

        Normalize(&b[3]);

        b[6] = Evector[2][0] * rot[0] + 
        Evector[2][1] * rot[3] + 
        Evector[2][2] * rot[6];
        b[7] = Evector[2][0] * rot[1] + 
        Evector[2][1] * rot[4] + 
        Evector[2][2] * rot[7];
        b[8] = Evector[2][0] * rot[2] + 
        Evector[2][1] * rot[5] + 
        Evector[2][2] * rot[8];

        Normalize(&b[6]);

        /*
         *  b3 = b1 x b2 
         */
        /*VOP_3D_COORDS_CROSS_PRODUCT(cp[0], cp[1], cp[2],
                                      b[0],   b[1],  b[2],
                                      b[3],   b[4],  b[5]);*/
        cp[0] = (b[1] * b[5]) - (b[2] * b[4]);
        cp[1] = (b[2] * b[3]) - (b[0] * b[5]);
        cp[2] = (b[0] * b[4]) - (b[1] * b[3]);

        if((cp[0] * b[6] + cp[1] * b[7] + cp[2] * b[8]) < 0.0f)
            sig3 = -1.0f;
        else
            sig3 = 1.0f;

        b[6] = cp[0]; 
        b[7] = cp[1]; 
        b[8] = cp[2];

        /*
         *  U has the best rotation 
         */
        for(k=k3=0; k<3; k++,k3+=3)
            for(i=i3=0;i<3; i++,i3+=3)
                for(j=0; j<3; j++) 
                {
                    U[i3 + j] += Evector[k][j] * b[k3 + i];
                }

        /*
         *  E = E0 - sqrt(mu1) - sqrt(mu2) - sig3*sqrt(mu3) 
         */
        rms_return = mwss - sqrt(fabs(Eigenvalue[0])) - sqrt(fabs(Eigenvalue[1])) - sig3 * sqrt(fabs(Eigenvalue[2]));
        if(rms_return < 0.0f)
        {
            rms_return = 0.0f;
        } 
        else 
        {
            rms_return = sqrt( (2.0f * rms_return) / total_mass);
        }


        /*
         *  Move the reference back so that it stays unchanged. This is
         *  necessary to preserve the meaning of CM shift on next frame
         *  iteration. 
         */
        for(k = 0; k < n; k++) 
        {
            Vec[3*k] += (float)cofmX1;
            Vec[3*k+1] += (float)cofmY1;
            Vec[3*k+2] += (float)cofmZ1;
        }

        if(mode == 2) 
        {
            /*
             *  Save rotation matrix which does the best overlap of trajectory
             *  coordinates to reference coordinates when they are both centered
             *  on their CMs. The actual modification (=rotation) of trajectory
             *  coords happens in the calling routine (actions.c::transformRMS())
             */
            rotation[0][0] = (float)U[0];
            rotation[0][1] = (float)U[1];
            rotation[0][2] = (float)U[2];
            rotation[1][0] = (float)U[3];
            rotation[1][1] = (float)U[4];
            rotation[1][2] = (float)U[5];
            rotation[2][0] = (float)U[6];
            rotation[2][1] = (float)U[7];
            rotation[2][2] = (float)U[8];

            /*
             *  Once the reference coords are shifted back to its original
             *  position (the for-cycle above), we need to shift trajectory
             *  coordinates by the same amount (i.e. CM of the reference) 
             *  to get them overlapped with the reference. The actual
             *  translation of trajectory coordinates happens in the calling
             *  routine (actions.c::transformRMS() )
             */
            translation[0] = (float)cofmX1;
            translation[1] = (float)cofmY1;
            translation[2] = (float)cofmZ1;

            /* First apply the rotation (which was calculated for both 
            trajectory and reference coords shifted to their CMs). The
            order (first rotation, then translation) is important.*/
            for (k=0; k < n; k++) 
            {
                /*VOP_3x3_TIMES_COORDS(rotation, toFitX[k], toFitY[k], toFitZ[k], xtemp, ytemp, ztemp);*/
                xtemp = rotation[0][0] * toFitVec[3*k] +  rotation[0][1] * toFitVec[3*k+1] +  rotation[0][2] * toFitVec[3*k+2];
                ytemp = rotation[1][0] * toFitVec[3*k] +  rotation[1][1] * toFitVec[3*k+1] +  rotation[1][2] * toFitVec[3*k+2];
                ztemp = rotation[2][0] * toFitVec[3*k] +  rotation[2][1] * toFitVec[3*k+1] +  rotation[2][2] * toFitVec[3*k+2];
                toFitVec[3*k] = xtemp;
                toFitVec[3*k+1] = ytemp;
                toFitVec[3*k+2] = ztemp;

                toFitVec[3*k] += (float)cofmX1;
                toFitVec[3*k+1] += (float)cofmY1;
                toFitVec[3*k+2] += (float)cofmZ1;
            }
        } 
        else if(mode == 1)
        {
            /* Nothing. XYZ moved back. ToFitXYZ moved to (0,0,0) */
        }
        else if(mode == 0)
        {
            /* Or just move them back to their original position */
            for(k = 0; k < n; k++) 
            {
                toFitVec[3*k] += (float)cofmX;
                toFitVec[3*k+1] += (float)cofmY;
                toFitVec[3*k+2] += (float)cofmZ;
            }
        } 
    } 
    else
    {
        ierr = -1;
    }

    if (ierr != 0) 
    {
        switch (ierr) 
        {
            case -1: err = "Number of atoms less than 2"; break;
            case -2: /* ierr is never set to -2 previously ?? */
                     err = "Illegal weights"; break;
            default: err = "Unknown error"; break;
        }
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "RMS: CalculateRMS - error: %s\n", err);
    }
    delete []weights;
    return (float) rms_return;
}
