/*
 * misc/BezierCurvesListDataCall.cpp
 *
 * Copyright (C) 2013 by TU Dresden.
 * Alle Rechte vorbehalten.
 */

#include "geometry_calls/BezierCurvesListDataCall.h"

namespace megamol::geocalls {


/*
 * BezierCurvesListDataCall::Curves::Curves
 */
BezierCurvesListDataCall::Curves::Curves()
        : layout(DATALAYOUT_NONE)
        , data(NULL)
        , data_memory_ownership(false)
        , data_cnt(0)
        , index(NULL)
        , index_memory_ownership(false)
        , index_cnt(0)
        , rad(0.5f) {
    this->col[0] = this->col[1] = this->col[2] = 128;
}


/*
 * BezierCurvesListDataCall::Curves::Curves
 */
BezierCurvesListDataCall::Curves::Curves(const BezierCurvesListDataCall::Curves& src)
        : layout(DATALAYOUT_NONE)
        , data(NULL)
        , data_memory_ownership(false)
        , data_cnt(0)
        , index(NULL)
        , index_memory_ownership(false)
        , index_cnt(0)
        , rad(0.5f) {
    // not very nice, but the best way, I believe
    this->Set(const_cast<Curves&>(src), true);
}


/*
 * BezierCurvesListDataCall::Curves::~Curves
 */
BezierCurvesListDataCall::Curves::~Curves() {
    this->Clear();
}


/*
 * BezierCurvesListDataCall::Curves::Curves
 */
void BezierCurvesListDataCall::Curves::Clear() {
    this->layout = DATALAYOUT_NONE;
    if (this->data != NULL) {
        if (this->data_memory_ownership) {
            delete[] this->data;
        }
        this->data = NULL;
    }
    this->data_cnt = 0;
    if (this->index != NULL) {
        if (this->index_memory_ownership) {
            delete[] this->index;
        }
        this->index_memory_ownership = NULL;
    }
    this->index_cnt = 0;
    // the last two are not clear, but I like to restore the whole default
    // values
    this->rad = 0.5f;
    this->col[0] = this->col[1] = this->col[2] = 128;
}


/*
 * BezierCurvesListDataCall::BezierCurvesListDataCall
 */
BezierCurvesListDataCall::BezierCurvesListDataCall()
        : AbstractGetData3DCall()
        , curves()
        , count()
        , static_indices(false) {
    // intentionally empty
}


/*
 * BezierCurvesListDataCallBezierCurvesListDataCallBezierDataCall
 */
BezierCurvesListDataCall::~BezierCurvesListDataCall() {
    this->Unlock();
    this->curves = NULL;
    this->count = 0;
    this->static_indices = false;
}
} // namespace megamol::geocalls
