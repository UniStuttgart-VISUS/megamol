/******************************************************************************/
/* File:             HD.cpp                                                   */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     19.06.2015                                               */
/*                                                                            */
/* Contains functions that compute the exact halfspace depth of a point z     */
/* w.r.t. n data points x[1],...x[n].                                         */
/*                                                                            */
/******************************************************************************/

#define _USE_MATH_DEFINES
#include "HD.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

namespace HalfspaceDepth {

/* Definition of constants */
const double eps_HD1 = 1e-8;
const double eps_HD2 = 1e-8;
const double eps_HDx = 1e-8;
const double eps_pivot = 1e-8;

/****************************************************************************/
/*                                                                          */
/* 'norm2' computes the Euclidean norm of a vector x in d-space.            */
/*                                                                          */
/****************************************************************************/

double norm2(double* x, int d) {
    double result = 0;
    for (int i = 0; i < d; i++) result += x[i] * x[i];
    return sqrt(result);
}

/****************************************************************************/
/*                                                                          */
/* 'intHD1' computes the integer hafspace depth of 0 w.r.t n data points    */
/* in R^1.                                                                  */
/*                                                                          */
/****************************************************************************/

int intHD1(double** x, int n) {
    int cnt1 = 0, cnt2 = 0;
    for (int i = 0; i < n; i++, x++) {
        if (**x < eps_HD1) cnt1++;
        if (**x > -eps_HD1) cnt2++;
    }
    return min(cnt1, cnt2);
}

/****************************************************************************/
/*                                                                          */
/* 'intHD2' computes the integer hafspace depth of 0 w.r.t n data points    */
/* in R^2.                                                                  */
/*                                                                          */
/* This is an implemetation of the algorithm of                             */
/* Rousseeuw, P.J.and Ruts, I. (1996). Algorithm AS 307: bivariate          */
/* location depth. Journal of the Royal Statistical Society. Series C:      */
/* Applied Statistics 45, 516–526.                                          */
/*                                                                          */
/****************************************************************************/

int intHD2(double** x, int n) {
    double* alpha = new double[n];
    int nt = 0; // how many zeros in in x ?
    int nh = 0; // how many points in the halfspace ?
    // compute angles alpha[i] and initilize array angle
    for (int i = 0; i < n; i++) {
        if (hypot(x[i][0], x[i][1]) <= eps_HD2)
            nt++;
        else {
            alpha[i - nt] = atan2(x[i][1], x[i][0]); // alpha in (-pi,pi]
            // correction for points like (-1, -1e-16)
            if (alpha[i - nt] < -M_PI + eps_HD2) alpha[i - nt] = M_PI;
            if (alpha[i - nt] <= eps_HD2) nh++;
        }
    }
    int nn = n - nt;
    // sort angles
    sort(alpha, alpha + nn);
    // compute halfspace depth
    int result = nh;
    if (result > 0) {
        int j = nh;
        for (int i = 0; i < nh; i++) {
            while ((j <= nn - 1) && (alpha[j] - M_PI <= alpha[i] + eps_HD2)) j++;
            if (j - i <= result) result = j - i - 1;
        }
        j = 0;
        for (int i = nh; i < nn; i++) {
            while ((j <= nh - 1) && (alpha[j] + M_PI <= alpha[i] + eps_HD2)) j++;
            if (j - (i - nn) <= result) result = j - (i - nn) - 1;
        }
    }
    delete[] alpha;
    return result + nt;
}

/****************************************************************************/
/*                                                                          */
/* 'nHD_Rec' computes the integer halfspace depth of 0 w.r.t n data points  */
/* in R^d.                                                                  */
/*                                                                          */
/* 'nHD_Rec' implements the recursive algorithm (k = 1) as described in     */
/* Section 3.3 of "Exact computation of the halfspace depth" by             */
/* Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)               */
/*                                                                          */
/****************************************************************************/

int nHD_Rec(double** xx, int n, int d) {
    if (d == 1) return intHD1(xx, n);
    if (d == 2) return intHD2(xx, n);

    double* y = new double[d - 1];
    double* z = new double[d];
    double** x = new double*[n];
    for (int k = 0; k < n; k++) x[k] = new double[d - 1];

    int result = n;
    for (int i = 0; i < n; i++) {

        int kmax = d;
        double xmax = 0;
        for (int k = 0; k < d; k++)
            if (fabs(xx[i][k]) > xmax) {
                xmax = fabs(xx[i][k]);
                kmax = k;
            }
        if (xmax > eps_HDx) {
            int nNull = 0, nPos = 0, nNeg = 0, m = 0;
            for (int k = 0; k < d; k++) z[k] = xx[i][k] / xx[i][kmax];
            // project data points
            for (int j = 0; j < n; j++) {
                double alpha = -xx[j][kmax];
                for (int k = 0; k < kmax; k++) y[k] = xx[j][k] + alpha * z[k];
                for (int k = kmax; k < d - 1; k++) y[k] = xx[j][k + 1] + alpha * z[k + 1];
                if (norm2(y, d - 1) > eps_HDx) {
                    for (int k = 0; k < d - 1; k++) x[m][k] = y[k];
                    m++;
                } else {
                    // in this case alpha = -sign(x_i*x_j)
                    if (alpha > eps_HDx)
                        nPos++;
                    else if (alpha < -eps_HDx)
                        nNeg++;
                    else
                        nNull++;
                }
            }
            result = min(result, nHD_Rec(x, m, d - 1) + nNull + min(nPos, nNeg));
            if (result == 0) break;
        }
    }
    for (int k = 0; k < n; k++) delete[] x[k];
    delete[] x;
    delete[] z;
    delete[] y;
    return result;
}

/****************************************************************************/
/*                                                                          */
/* 'HD_Rec' computes the halfspace depth of a point z w.r.t n data          */
/* points in R^d.                                                           */
/*                                                                          */
/* 'HD_Rec' does some preprocessing of the data and then calls              */
/* 'nHD_Rec'.                                                               */
/*                                                                          */
/* See also the description in the header file.                             */
/*                                                                          */
/* HD_Rec calls the following routines:                                     */
/*     norm2                                                                */
/*     intHD1                                                               */
/*     intHD2                                                               */
/*     nHD_Rec                                                              */
/*                                                                          */
/****************************************************************************/

double HD_Rec(double* z, double** xx, int n, int d) {
    if (n <= 0) throw invalid_argument("n <= 0");
    if (d <= 0) throw invalid_argument("d <= 0");
    // preprocess data
    // subtract z from all data points x[i]
    int m = 0;
    double** x = new double*[n];
    for (int i = 0; i < n; i++) {
        x[m] = new double[d];
        for (int j = 0; j < d; j++) x[m][j] = xx[i][j] - z[j];
        if (norm2(x[m], d) >= eps_HDx)
            m++;
        else
            delete[] x[m];
    }
    int result = nHD_Rec(x, m, d) + (n - m);
    // deallocate array x
    for (int i = 0; i < m; i++) delete[] x[i];
    delete[] x;
    return result / (double)n;
}

/****************************************************************************/
/* 'getRank' computes the rank of the matrix x                              */
/*                                                                          */
/* 'getRank' is used in preprocessing the data in HD_comb and HD_Comb2.     */
/* 'getRank' detects if the data points are contained in a lower            */
/*  dimensional space, by computing the rank of the matrix formed by the    */
/*  data points x[i].                                                       */
/*                                                                          */
/****************************************************************************/

int getRank(double** x, int n, int d, int* piv) {
    int imax;
    int pc = 0, rank = 0;
    double amax;
    // copy x to A
    double** A = new double*[d];
    for (int i = 0; i < d; i++) {
        A[i] = new double[n];
        for (int j = 0; j < n; j++) A[i][j] = x[j][i];
    }
    rank = 0;
    for (int k = 0; k < min(n, d); k++) {
        // k-th elimination step
        do {
            imax = k;
            amax = fabs(A[k][pc]);
            // find maximum element in column
            for (int i = k + 1; i < d; i++) {
                if (fabs(A[i][pc]) > amax) {
                    amax = fabs(A[i][pc]);
                    imax = i;
                }
            }
            if (amax < eps_pivot) pc++;
        } while ((amax < eps_pivot) && (pc < n));
        if (pc < n) {
            rank++;
            piv[k] = pc;
            // exchange rows
            if (imax != k) {
                for (int j = pc; j < n; j++) {
                    double tmp = A[k][j];
                    A[k][j] = A[imax][j];
                    A[imax][j] = tmp;
                }
            }
            // elimination
            for (int i = k + 1; i < d; i++) {
                double factor = A[i][pc] / A[k][pc];
                for (int j = pc + 1; j < n; j++) A[i][j] -= factor * A[k][j];
            }
            if (++pc >= n) break;
        } else
            break;
    }
    for (int i = 0; i < d; i++) delete[] A[i];
    delete[] A;

    return rank;
}

/****************************************************************************/
/* 'project' projects the data points on a lower dimensional subspace.      */
/*                                                                          */
/* 'project' is used in preprocessing the data in HD_comb and HD_Comb2.     */
/* If the data points x[i] are contained in a subspace of dimension 'rank'  */
/* (as detected by a call to getRank), the representation of the data       */
/* points w.r.t. a basis of this subspace is computed. This gives a         */
/* representation of the data points in the Euclidean space of dimension    */
/* rank.                                                                    */
/*                                                                          */
/****************************************************************************/

void project(double** x, int n, int d, int rank, int indices[]) {
    double** z = new double*[n];
    for (int k = 0; k < n; k++) {
        z[k] = new double[rank];
        for (int i = 0; i < rank; i++) {
            z[k][i] = 0;
            for (int l = 0; l < d; l++) z[k][i] += x[k][l] * x[indices[i]][l];
        }
    }
    for (int k = 0; k < n; k++) {
        delete[] x[k];
        x[k] = z[k];
    }
    delete[] z;
}

/****************************************************************************/
/*                                                                          */
/* 'getNormal' computes the normal vector to the d-1 vectors passed         */
/* in A.                                                                    */
/*                                                                          */
/* If the rank of A is equal to d-1, then the function returns 'true' and   */
/* the normal vector is passed to the calling routine in 'normal[]'.        */
/* If the rank of A is less than d-1, then the function returns 'false'     */
/* and the value of 'normal[]' is undefined.                                */
/*                                                                          */
/****************************************************************************/

bool getNormal(double** A, int d, double* normal) {
    int imax, jmax;
    int* colp = new int[d];
    double amax;
    for (int k = 0; k < d - 1; k++) {
        imax = k;
        amax = fabs(A[k][k]);
        colp[k] = k;
        // find maximum element in column
        for (int i = k + 1; i < d - 1; i++) {
            if (fabs(A[i][k]) > amax) {
                amax = fabs(A[i][k]);
                imax = i;
            }
        }
        // maximum eual to zero => complete pivoting
        if (amax < eps_pivot) {
            for (int j = k + 1; j < d; j++) {
                for (int i = k; i < d - 1; i++) {
                    if (fabs(A[i][j]) > amax) {
                        amax = fabs(A[i][j]);
                        imax = i;
                        jmax = j;
                    }
                }
            }
            if (amax < eps_pivot) {
                delete[] colp;
                return false;
            }
            // exchange columns
            for (int i = 0; i < d - 1; i++) {
                double tmp = A[i][k];
                A[i][k] = A[i][jmax];
                A[i][jmax] = tmp;
            }
            colp[k] = jmax;
        }
        // exchange rows
        if (imax != k) {
            for (int j = k; j < d; j++) {
                double tmp = A[k][j];
                A[k][j] = A[imax][j];
                A[imax][j] = tmp;
            }
        }
        // elimination
        for (int i = k + 1; i < d - 1; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < d; j++) A[i][j] -= factor * A[k][j];
        }
    }
    // back substitution
    colp[d - 1] = d - 1;
    normal[d - 1] = -1;
    for (int k = d - 2; k >= 0; k--) {
        normal[k] = A[k][d - 1] / A[k][k];
        for (int i = k - 1; i >= 0; i--) A[i][d - 1] -= normal[k] * A[i][k];
    }
    // reverse column permutations
    for (int k = d - 1; k >= 0; k--) {
        if (colp[k] != k) {
            double temp = normal[k];
            normal[k] = normal[colp[k]];
            normal[colp[k]] = temp;
        }
    }
    delete[] colp;

    return true;
}

int nHD_Comb(double** xx, int n, int d);

/****************************************************************************/
/*                                                                          */
/* 'HD1proj' performs the following steps:                                  */
/*                                                                          */
/*  1) All data points x[i] are projected in the direction p,               */
/*     i.e., z[i] = p'x[i] is computed.                                     */
/*  2) The univariate integer halfspace depth of 0 is computed w.r.t. all   */
/*     the z[i] that are not equal to 0.                                    */
/*  3) If there are more than d-1 values z[i] that are equal to zero,       */
/*     the respective points are projected on the orthogonal complement     */
/*     of p. Then, the integer halfspace depth of 0 w.r.t. these            */
/*     projected points is computed.                                        */
/*  4) The sum of the values from step 2) and 3) is returned.               */
/*                                                                          */
/****************************************************************************/

int HD1proj(double** x, int n, int d, double* p, int indices[]) {
    int cnt0 = 0, cnt1 = 0, cnt2 = 0, HDproj = 0;
    int* plane = new int[n];
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < d; j++) sum += p[j] * x[i][j];
        if (sum > eps_HD1)
            cnt1++;
        else if (sum < -eps_HD1)
            cnt2++;
        else
            plane[cnt0++] = i;
    }
    if (cnt0 > d - 1) {
        // recursion
        double** z = new double*[cnt0];
        for (int i = 0; i < cnt0; i++) {
            z[i] = new double[d - 1];
            for (int j = 0; j < d - 1; j++) {
                z[i][j] = 0;
                for (int k = 0; k < d; k++) z[i][j] += x[indices[j]][k] * x[plane[i]][k];
            }
        }
        HDproj = nHD_Comb(z, cnt0, d - 1);
        for (int i = 0; i < cnt0; i++) delete[] z[i];
        delete[] z;
    }
    delete[] plane;
    return min(cnt1, cnt2) + HDproj;
}

/****************************************************************************/
/*                                                                          */
/* 'nHD_Comb' computes the integer halfspace depth of 0 w.r.t n data points */
/* in R^d.                                                                  */
/*                                                                          */
/* 'nHD_Comb' implements the combinatorial algorithm (k = d-1) as described */
/* in Section 3.1 of "Exact computation of the halfspace depth" by          */
/* Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)               */
/*                                                                          */
/****************************************************************************/

int nHD_Comb(double** xx, int n, int d) {
    if (d == 1) return intHD1(xx, n);
    if (d == 2) return intHD2(xx, n);

    int result = n + 1;
    double** a = new double*[d - 1];
    for (int i = 0; i < d - 1; i++) a[i] = new double[d];
    double* p = new double[d];
    int* indices = new int[d - 1];
    indices[0] = -1;
    int pos = 0;
    while (pos >= 0) {
        indices[pos]++;
        for (pos++; pos < d - 1; pos++) indices[pos] = indices[pos - 1] + 1;
        pos--;
        do {
            for (int i = 0; i < d - 1; i++)
                for (int j = 0; j < d; j++) a[i][j] = xx[indices[i]][j];
            if (getNormal(a, d, p)) result = min(result, HD1proj(xx, n, d, p, indices));
            indices[pos]++;
        } while (indices[pos] < n - d + pos + 2);
        do
            pos--;
        while (pos >= 0 && indices[pos] >= n - d + pos + 1);
    }
    for (int i = 0; i < d - 1; i++) delete[] a[i];
    delete[] a;
    delete[] p;
    delete[] indices;
    return result;
}

/****************************************************************************/
/*                                                                          */
/* 'HD_Comb' computes the halfspace depth of a point z w.r.t n data         */
/* points in R^d.                                                           */
/*                                                                          */
/* 'HD_Comb' does some preprocessing of the data and then calls             */
/* 'nHD_Comb'.                                                              */
/*                                                                          */
/* See also the description in the header file.                             */
/*                                                                          */
/* HD_Comb calls the following routines:                                    */
/*     norm2                                                                */
/*     intHD1                                                               */
/*     intHD2                                                               */
/*     getRank                                                              */
/*     project                                                              */
/*     getNormal                                                            */
/*     HD1proj                                                              */
/*     nHD_Rec                                                              */
/*                                                                          */
/****************************************************************************/

double HD_Comb(double* z, double** xx, int n, int d) {
    if (n <= 0) throw invalid_argument("n <= 0");
    if (d <= 0) throw invalid_argument("d <= 0");
    // preprocess data
    //   subtract z from all data points x[i]
    //   check whether the data points are concentrated on a lower
    //   dimensional spcae
    int m = 0, rank;
    int* indices = new int[d];
    double** x = new double*[n];
    for (int i = 0; i < n; i++) {
        x[m] = new double[d];
        for (int j = 0; j < d; j++) x[m][j] = xx[i][j] - z[j];
        if (norm2(x[m], d) >= eps_HDx)
            m++;
        else
            delete[] x[m];
    }
    if (m == 0) return 1.0;

    rank = getRank(x, m, d, indices);
    if (rank < d) project(x, m, d, rank, indices);
    int result = nHD_Comb(x, m, rank) + (n - m);
    // deallocate array x
    for (int i = 0; i < m; i++) delete[] x[i];
    delete[] x;
    delete[] indices;
    return result / (double)n;
}

/****************************************************************************/
/*                                                                          */
/* 'getBasisComplement' computes a basis of the orthogonal complement of    */
/* the d-2 vectors passed in A.                                             */
/*                                                                          */
/* If the rank of A is equal to d-2, then the function returns 'true' and   */
/* the two basis vectors are passed to the calling routine in 'basis[][]'.  */
/* If the rank of A is less than d-2, then the function returns 'false'     */
/* and the value of 'basis[]' is undefined.                                 */
/*                                                                          */
/****************************************************************************/

bool getBasisComplement(double** A, int d, double** basis) {
    int imax, jmax;
    int* colp = new int[d];
    double amax;
    for (int k = 0; k < d - 2; k++) {
        imax = k;
        amax = fabs(A[k][k]);
        colp[k] = k;
        // find maximum element in column
        for (int i = k + 1; i < d - 2; i++) {
            if (fabs(A[i][k]) > amax) {
                amax = fabs(A[i][k]);
                imax = i;
            }
        }
        // maximum equla to zero  => complete pivoting
        if (amax < eps_pivot) {
            for (int j = k + 1; j < d; j++) {
                for (int i = k; i < d - 2; i++) {
                    if (fabs(A[i][j]) > amax) {
                        amax = fabs(A[i][j]);
                        imax = i;
                        jmax = j;
                    }
                }
            }
            if (amax < eps_pivot) {
                delete[] colp;
                return false;
            }
            // exchange columns
            for (int i = 0; i < d - 2; i++) {
                double tmp = A[i][k];
                A[i][k] = A[i][jmax];
                A[i][jmax] = tmp;
            }
            colp[k] = jmax;
        }
        // exchange rows
        if (imax != k) {
            for (int j = k; j < d; j++) {
                double tmp = A[k][j];
                A[k][j] = A[imax][j];
                A[imax][j] = tmp;
            }
        }
        // elimination
        for (int i = k + 1; i < d - 2; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < d; j++) A[i][j] -= factor * A[k][j];
        }
    }
    // back substitution
    colp[d - 2] = d - 2;
    colp[d - 1] = d - 1;
    basis[0][d - 2] = -1;
    basis[0][d - 1] = 0;
    basis[1][d - 2] = 0;
    basis[1][d - 1] = -1;
    for (int k = d - 3; k >= 0; k--) {
        basis[0][k] = A[k][d - 2] / A[k][k];
        basis[1][k] = A[k][d - 1] / A[k][k];
        for (int i = k - 1; i >= 0; i--) {
            A[i][d - 2] -= basis[0][k] * A[i][k];
            A[i][d - 1] -= basis[1][k] * A[i][k];
        }
    }
    // reverse column permutations
    for (int k = d - 1; k >= 0; k--) {
        if (colp[k] != k) {
            for (int l = 0; l < 2; l++) {
                double temp = basis[l][k];
                basis[l][k] = basis[l][colp[k]];
                basis[l][colp[k]] = temp;
            }
        }
    }
    delete[] colp;
    return true;
}

int nHD_Comb2(double** xx, int n, int d);

/****************************************************************************/
/*                                                                          */
/* 'HD2proj' performs the following steps:                                  */
/*                                                                          */
/*  1) All data points x[i] are projected on the space spanned by the       */
/*     two vectors passed in p, i.e., y[i,1] = p[1]'x[i] and                */
/*     y[i,2] = p[2]'x[i] are computed.                                     */
/*  2) The bivariate integer halfspace depth of 0 is computed w.r.t. all    */
/*     the y[i] that are not equal to (0,0).                                */
/*  3) If there are more than d-2 values y[i] that are equal to (0,0),      */
/*     the respective points are projected on the orthogonal complement     */
/*     of p. Then, the integer halfspace depth of 0 w.r.t. these            */
/*     projected points is computed.                                        */
/*  4) The sum of the values from step 2) and 3) is returned.               */
/*                                                                          */
/****************************************************************************/

int HD2proj(double** x, int n, int d, double** p, int* indices) {
    double** y = new double*[n];
    for (int i = 0; i < n; i++) y[i] = new double[2];

    int cnt0 = 0, cnt1 = 0, HDproj = 0;
    int* plane = new int[n];
    for (int i = 0; i < n; i++) {
        y[cnt1][0] = y[cnt1][1] = 0;
        for (int j = 0; j < d; j++)
            for (int k = 0; k < 2; k++) y[cnt1][k] += p[k][j] * x[i][j];
        if (norm2(y[cnt1], 2) > eps_HD2)
            cnt1++;
        else
            plane[cnt0++] = i;
    }
    if (cnt0 > d - 2) {
        double** z = new double*[cnt0];
        for (int i = 0; i < cnt0; i++) {
            z[i] = new double[d - 2];
            for (int j = 0; j < d - 2; j++) {
                z[i][j] = 0;
                for (int k = 0; k < d; k++) z[i][j] += x[indices[j]][k] * x[plane[i]][k];
            }
        }
        HDproj = nHD_Comb2(z, cnt0, d - 2);
        for (int i = 0; i < cnt0; i++) delete[] z[i];
        delete[] z;
    }
    int result = intHD2(y, cnt1) + HDproj;
    delete[] plane;
    for (int i = 0; i < n; i++) delete[] y[i];
    delete[] y;
    return result;
}

/****************************************************************************/
/*                                                                          */
/* 'nHD_Comb2' computes the integer halfspace depth of 0 w.r.t n data       */
/* points in R^d.                                                           */
/*                                                                          */
/* 'nHD_Comb2' implements the combinatorial algorithm (k = d-2) as          */
/* described in Section 3.2 of "Exact computation of the halfspace depth"   */
/* by Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)            */
/*                                                                          */
/****************************************************************************/

int nHD_Comb2(double** xx, int n, int d) {
    if (d == 1) return intHD1(xx, n);
    if (d == 2) return intHD2(xx, n);

    int result = n + 1;
    double** a = new double*[d - 2];
    for (int i = 0; i < d - 2; i++) a[i] = new double[d];
    double** p = new double*[2];
    for (int i = 0; i < 2; i++) p[i] = new double[d];
    int* indices = new int[d - 2];

    indices[0] = -1;
    int pos = 0;
    while (pos >= 0) {
        indices[pos]++;
        for (pos++; pos < d - 2; pos++) indices[pos] = indices[pos - 1] + 1;
        pos--;
        do {
            for (int i = 0; i < d - 2; i++)
                for (int j = 0; j < d; j++) a[i][j] = xx[indices[i]][j];
            if (getBasisComplement(a, d, p)) result = min(result, HD2proj(xx, n, d, p, indices));
            indices[pos]++;
        } while (indices[pos] < n - d + pos + 3);
        do
            pos--;
        while (pos >= 0 && indices[pos] >= n - d + pos + 2);
    }
    for (int i = 0; i < d - 2; i++) delete[] a[i];
    delete[] a;
    for (int i = 0; i < 2; i++) delete[] p[i];
    delete[] p;
    delete[] indices;
    return result;
}

/****************************************************************************/
/*                                                                          */
/* 'HD_Comb2' computes the halfspace depth of a point z w.r.t. n data       */
/* points in R^d.                                                           */
/*                                                                          */
/* 'HD_Comb2' does some preprocessing of the data and then calls            */
/* 'nHD_Comb2'.                                                             */
/*                                                                          */
/* See also the description in the header file.                             */
/*                                                                          */
/* HD_Comb2 calls the following routines:                                   */
/*     norm2                                                                */
/*     intHD1                                                               */
/*     intHD2                                                               */
/*     getRank                                                              */
/*     project                                                              */
/*     getBasisComplement                                                   */
/*     HD2proj                                                              */
/*     nHD_Rec                                                              */
/*                                                                          */
/****************************************************************************/

double HD_Comb2(double* z, double** xx, int n, int d) {
    if (n <= 0) throw invalid_argument("n <= 0");
    if (d <= 0) throw invalid_argument("d <= 0");
    int m = 0, rank;
    int* indices = new int[d];
    double** x = new double*[n];
    // preprocess data
    //   subtract z from all data points x[i]
    //   check whether the data points are concentrated on a lower
    //   dimensional spcae
    for (int i = 0; i < n; i++) {
        x[m] = new double[d];
        for (int j = 0; j < d; j++) x[m][j] = xx[i][j] - z[j];
        if (norm2(x[m], d) >= eps_HDx)
            m++;
        else
            delete[] x[m];
    }
    if (m == 0) return 1.0;
    rank = getRank(x, m, d, indices);
    if (rank < d) project(x, m, d, rank, indices);

    int result = nHD_Comb2(x, m, rank) + (n - m);
    // deallocate array x
    for (int i = 0; i < m; i++) delete[] x[i];
    delete[] x;
    delete[] indices;
    return result;
    return result / (double)n;
}

} // namespace HalfspaceDepth
