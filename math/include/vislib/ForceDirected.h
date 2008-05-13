/*
 * ForceDirected.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FORCEDIRECTED_H_INCLUDED
#define VISLIB_FORCEDIRECTED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <stdlib.h>
#include <time.h>

#include "vislib/Array.h"
#include "vislib/Vector.h"

namespace vislib {
namespace math {

    /**
     * Basic force-directed graph layout.
     * Uses Coulomb repulsion and Hooke attraction to push nodes around until the system
     * energy is lower than maxEnergy or a maximum number of iterations have been done.
     * Current implementation costs O(N^3), because it is said that N iterations are needed.
     *
     * TI must support a float Weight(TI &other) method for the edge weight and
     * float TotalWeight() for the total weight of edges connected to an instance of TI.
     *
     * Increasing the number of items in the input data AFTER calling init will result in
     * the instance running away with your girlfriend, so just refrain from doing that.
     *
     * Attraction and repulsion scaling factors are currently unused because the alternative
     * Implementation with the CCVisu generic forces behaves weirdly.
     */
    template<class TI, class TO, unsigned int D>
    class ForceDirected {

        /** Output Element Type */
        typedef Vector<TO, D> OutElement;

        /** Result Type (reduced dimensionality) */
        typedef Array<Vector<TO, D> > ResultType;

    public:
        /**
         * Initialize parameters for layout.
         *
         * @param inputData Source data.
         * @param[out] out Layout result. Must be pre-allocated and will be updated by Compute and SingleStep.
         * @param maximumIterations maximum number of iterations to perform regardless of remaining system energy.
         * @param step integration step.
         * @param attraction attraction factor for CCVisu (currently unused)
         * @param repulsion repulsion factor for CCVisu (currently unused)
         * @param maximumEnergy energy threshold of system for the iteration to stop.
         * @param damping damping factor applied to forces for quicker becalming of system.
         */
        ForceDirected(Array<TI> &inputData, ResultType &out, unsigned int maximumIterations, float step, 
                    float attraction, float repulsion, float maximumEnergy, float damping) : inData(inputData), outData(out) {
            this->maxIterations = maximumIterations;
            currVelocity = new ResultType;
            attractionFactor = attraction;
            repulsionFactor = repulsion;
            stepLen = step;
            maxEnergy = maximumEnergy;
            this->damping = damping;
            //this->Init();
        }

        /** dtor. */
        ~ForceDirected() {
            delete currVelocity;
        }

        /**
         * Iterate the force calculations until the energy is below maxEnergy or maxIterations have been performed.
         *
         * changes out.
         */
        void Compute(void);

        /**
         * Perform a single integration step.
         *
         * changes out.
         *
         * @return the resulting system energy.
         */
        float SingleStep(void);

        /**
         * Randomize initial positions of all nodes, zero velocities
         */
        void Init();

    private:
        /** Reference to source data */
        Array<TI> &inData;

        /** Reference to output data */
        ResultType &outData;

        /** current velocity of all object is stored here */
        ResultType *currVelocity;

        /** maximum number of iterations */
        unsigned int maxIterations;

        /** unused */
        float attractionFactor;

        /** unused */
        float repulsionFactor;

        /** integration step length */
        float stepLen;

        /** maximum energy threshold */
        float maxEnergy;

        /** force damping */
        float damping;
    };

    /*
     * vislib::math::ForceDirected<TI, TO, D>::Init
     */
    template<class TI, class TO, unsigned int D>
    void ForceDirected<TI, TO, D>::Init(void) {

        SIZE_T count = inData.Count();
        currVelocity->SetCount(count);
        srand((unsigned)time(NULL));

        for (SIZE_T i = 0; i < count; i++) {
            for (unsigned int j = 0; j < D; j++) {
                outData[i][j] = (float)(((float)rand() / (float)RAND_MAX) * (float)count * 2.0);
            }
            (*currVelocity)[i].SetNull();
        }
    }

    /*
     * vislib::math::ForceDirected<TI, TO, D>::SingleStep
     */
    template<class TI, class TO, unsigned int D>
    float ForceDirected<TI, TO, D>::SingleStep(void) {

        SIZE_T count = inData.Count();
        OutElement currForce;
        OutElement diff, normdiff;
        float energy = 0;
        float w, twi, twj, dl;
        //float k = 1; // in theory 1/(4Pi epsilon_0)

        for (SIZE_T i = 0; i < count; i++) {
            currForce.SetNull();
            twi = inData[i].TotalWeight();
            for (SIZE_T j = 0; j < count; j++) {
                if (i != j) {
                    //diff = outData[j] - outData[i];
                    diff = outData[i] - outData[j];
                    if ((dl = diff.Length()) > 0.0000001) {
                        normdiff = diff;
                        normdiff.Normalise();
                        w = inData[i].Weight(inData[j]);
                        twj = inData[j].TotalWeight();
                        // CCVisu stuff (meh?)
                        //currForce += normdiff * ((twi / attractionFactor) * pow(dl, attractionFactor));
                        //currForce += -normdiff * (pow(twi * twj, repulsionFactor) * log(dl));
                        
                        // coulomb repulsion
                        currForce += normdiff * 1.0 / diff.Length();
                        //currForce += normdiff * 1.0 / diff.Length();
                        
                        // hooke attraction scaled by connection weight
                        //currForce += -normdiff * w *(diff.Length() - 1.0f);
                        // rest length proportional to weight!
                        if (w > 0) {
                            currForce += -normdiff * w * (diff.Length() - 1.0f / w);
                        } else {
                            currForce += -normdiff * w * (diff.Length() - 1.0f);
                        }
                        //currForce += -normdiff * inData[i].Weight(inData[j]) *(diff.Length() - twi);
                        //currForce += k * ((inData[i].Weight(inData[j]) * inData[i].Weight(inData[j]))
                        //		/ (diff.Length() * diff.Length())) * normdiff;
                    }
                }
            }
            //(*tmpData)[i] += currForce;
            (*currVelocity)[i] = ((*currVelocity)[i] + stepLen * currForce) * damping;
            outData[i] = outData[i] + stepLen * (*currVelocity)[i];
            energy += (*currVelocity)[i].Length() * (*currVelocity)[i].Length();
        }
        return energy;
    }

    /*
     * vislib::math::ForceDirected<TI, TO, D>::Compute
     */
    template<class TI, class TO, unsigned int D>
    void ForceDirected<TI, TO, D>::Compute(void) {
        
        this->Init();

        unsigned int currIter = 0;
        while (this->SingleStep() > maxEnergy) {
            if (currIter++ > maxIterations) {
                return;
            }
        }
    }

} /* end namespace math */
} /* end namespace vislib */



#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FORCEDIRECTED_H_INCLUDED */
