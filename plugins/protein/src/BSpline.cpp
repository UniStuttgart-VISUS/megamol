/*
 * BSpline.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "protein/BSpline.h"

using namespace megamol;

/*
 * protein::BSpline::BSpline
 */
protein::BSpline::BSpline() {
    this->N = 0;

    this->B.SetAt(0, 0, -1.0f / 6.0f);
    this->B.SetAt(0, 1, 3.0f / 6.0f);
    this->B.SetAt(0, 2, -3.0f / 6.0f);
    this->B.SetAt(0, 3, 1.0f / 6.0f);
    this->B.SetAt(1, 0, 3.0f / 6.0f);
    this->B.SetAt(1, 1, -6.0f / 6.0f);
    this->B.SetAt(1, 2, 3.0f / 6.0f);
    this->B.SetAt(1, 3, 0.0f / 6.0f);
    this->B.SetAt(2, 0, -3.0f / 6.0f);
    this->B.SetAt(2, 1, 0.0f / 6.0f);
    this->B.SetAt(2, 2, 3.0f / 6.0f);
    this->B.SetAt(2, 3, 0.0f / 6.0f);
    this->B.SetAt(3, 0, 1.0f / 6.0f);
    this->B.SetAt(3, 1, 4.0f / 6.0f);
    this->B.SetAt(3, 2, 1.0f / 6.0f);
    this->B.SetAt(3, 3, 0.0f / 6.0f);
}

/*
 * protein::BSpline::~BSpline
 */
protein::BSpline::~BSpline() {}

/*
 * protein::BSpline::setG
 * set the coordinates for the geometry matrix G
 */
void protein::BSpline::setG(vislib::math::Vector<float, 3> v1, vislib::math::Vector<float, 3> v2,
    vislib::math::Vector<float, 3> v3, vislib::math::Vector<float, 3> v4) {
    this->G.SetAt(0, 0, v1[0]);
    this->G.SetAt(0, 1, v1[1]);
    this->G.SetAt(0, 2, v1[2]);
    this->G.SetAt(0, 3, 1.0f);
    this->G.SetAt(1, 0, v2[0]);
    this->G.SetAt(1, 1, v2[1]);
    this->G.SetAt(1, 2, v2[2]);
    this->G.SetAt(1, 3, 1.0f);
    this->G.SetAt(2, 0, v3[0]);
    this->G.SetAt(2, 1, v3[1]);
    this->G.SetAt(2, 2, v3[2]);
    this->G.SetAt(2, 3, 1.0f);
    this->G.SetAt(3, 0, v4[0]);
    this->G.SetAt(3, 1, v4[1]);
    this->G.SetAt(3, 2, v4[2]);
    this->G.SetAt(3, 3, 1.0f);
}

/*
 * protein::BSpline::setN
 * set the number of segments to create --> this function also sets matrix S!
 */
void protein::BSpline::setN(unsigned int n) {
    this->N = n;

    this->S.SetAt(0, 0, 6.0f / (float)pow((double)n, (double)3));
    this->S.SetAt(0, 1, 0.0f);
    this->S.SetAt(0, 2, 0.0f);
    this->S.SetAt(0, 3, 0.0f);
    this->S.SetAt(1, 0, 6.0f / (float)pow((double)n, (double)3));
    this->S.SetAt(1, 1, 2.0f / (float)pow((double)n, (double)2));
    this->S.SetAt(1, 2, 0.0f);
    this->S.SetAt(1, 3, 0.0f);
    this->S.SetAt(2, 0, 1.0f / (float)pow((double)n, (double)3));
    this->S.SetAt(2, 1, 1.0f / (float)pow((double)n, (double)2));
    this->S.SetAt(2, 2, 1.0f / (float)n);
    this->S.SetAt(2, 3, 0.0f);
    this->S.SetAt(3, 0, 0.0f);
    this->S.SetAt(3, 1, 0.0f);
    this->S.SetAt(3, 2, 0.0f);
    this->S.SetAt(3, 3, 1.0f);
}

/*
 * protein::BSpline::computeSpline
 * compute the spline from the given backbone coordinates
 */
bool protein::BSpline::computeSpline() {
    vislib::math::Vector<float, 3> tmpVec;

    if (this->N > 0 && this->backbone.size() > 4) {
        // clear the result vector
        this->result.clear();

        // START FIX

        // assign the geometry matrix
        this->setG(this->backbone.at(0), this->backbone.at(0), this->backbone.at(0), this->backbone.at(1));
        // compute the M matrix
        this->M = this->S * (this->B * this->G);
        // start spline segment computation
        for (unsigned int k = 0; k < this->N; k++) {
            for (unsigned int i = 3; i > 0; i--) {
                for (unsigned int j = 0; j < 4; j++) {
                    // M[i][j] = M[i][j] + M[i-1][j];
                    this->M.SetAt(i, j, this->M.GetAt(i, j) + this->M.GetAt(i - 1, j));
                }
            }
            // DrawTo( M[3][0]/M[3][3], M[3][1]/M[3][3], M[3][2]/M[3][3] );
            tmpVec.Set(this->M.GetAt(3, 0) / this->M.GetAt(3, 3), this->M.GetAt(3, 1) / this->M.GetAt(3, 3),
                this->M.GetAt(3, 2) / this->M.GetAt(3, 3));
            // add the computet vector to the result list
            this->result.push_back(tmpVec);
        }
        // end spline segment computation
        // assign the geometry matrix
        this->setG(this->backbone.at(0), this->backbone.at(0), this->backbone.at(1), this->backbone.at(2));
        // compute the M matrix
        this->M = this->S * (this->B * this->G);
        // start spline segment computation
        for (unsigned int k = 0; k < this->N; k++) {
            for (unsigned int i = 3; i > 0; i--) {
                for (unsigned int j = 0; j < 4; j++) {
                    // M[i][j] = M[i][j] + M[i-1][j];
                    this->M.SetAt(i, j, this->M.GetAt(i, j) + this->M.GetAt(i - 1, j));
                }
            }
            // DrawTo( M[3][0]/M[3][3], M[3][1]/M[3][3], M[3][2]/M[3][3] );
            tmpVec.Set(this->M.GetAt(3, 0) / this->M.GetAt(3, 3), this->M.GetAt(3, 1) / this->M.GetAt(3, 3),
                this->M.GetAt(3, 2) / this->M.GetAt(3, 3));
            // add the computet vector to the result list
            this->result.push_back(tmpVec);
        }
        // end spline segment computation

        // END FIX

        for (unsigned int index = 0; index < this->backbone.size() - 3; index++) {
            // assign the geometry matrix
            this->setG(this->backbone.at(index), this->backbone.at(index + 1), this->backbone.at(index + 2),
                this->backbone.at(index + 3));
            // compute the M matrix
            this->M = this->S * (this->B * this->G);
            // start spline segment computation
            for (unsigned int k = 0; k < this->N; k++) {
                for (unsigned int i = 3; i > 0; i--) {
                    for (unsigned int j = 0; j < 4; j++) {
                        // M[i][j] = M[i][j] + M[i-1][j];
                        this->M.SetAt(i, j, this->M.GetAt(i, j) + this->M.GetAt(i - 1, j));
                    }
                }
                // DrawTo( M[3][0]/M[3][3], M[3][1]/M[3][3], M[3][2]/M[3][3] );
                tmpVec.Set(this->M.GetAt(3, 0) / this->M.GetAt(3, 3), this->M.GetAt(3, 1) / this->M.GetAt(3, 3),
                    this->M.GetAt(3, 2) / this->M.GetAt(3, 3));
                // add the computet vector to the result list
                this->result.push_back(tmpVec);
            }
            // end spline segment computation
        }

        // FIX START
        unsigned int end = (unsigned int)this->backbone.size() - 1;

        // assign the geometry matrix
        this->setG(
            this->backbone.at(end - 2), this->backbone.at(end - 1), this->backbone.at(end), this->backbone.at(end));
        // compute the M matrix
        this->M = this->S * (this->B * this->G);
        // start spline segment computation
        for (unsigned int k = 0; k < this->N; k++) {
            for (unsigned int i = 3; i > 0; i--) {
                for (unsigned int j = 0; j < 4; j++) {
                    // M[i][j] = M[i][j] + M[i-1][j];
                    this->M.SetAt(i, j, this->M.GetAt(i, j) + this->M.GetAt(i - 1, j));
                }
            }
            // DrawTo( M[3][0]/M[3][3], M[3][1]/M[3][3], M[3][2]/M[3][3] );
            tmpVec.Set(this->M.GetAt(3, 0) / this->M.GetAt(3, 3), this->M.GetAt(3, 1) / this->M.GetAt(3, 3),
                this->M.GetAt(3, 2) / M.GetAt(3, 3));
            // add the computet vector to the result list
            this->result.push_back(tmpVec);
        }
        // end spline segment computation

        // END FIX

        return true;
    } else {
        // clear the result vector
        this->result.clear();
        return false;
    }
}
