/*
 * CallSpheres.h
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MOLSURFMAPCLUSTER_CLUSTERINGLOADER_CALL_INCLUDED
#define MOLSURFMAPCLUSTER_CLUSTERINGLOADER_CALL_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "HierarchicalClustering.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace molsurfmapcluster {

/**
 * Call for sphere data.
 */
class CallClusteringLoader : public core::AbstractGetData3DCall {
public:
    // TUTORIAL: These static const values should be set for each call that offers callback functionality.
    // This provides a guidance which callback function gets called.

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /**
     * Answer the name of the objects of this description.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallClusteringLoader"; }

    /**
     * Gets a human readable description of the module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call to get Clustering-Data"; }

    /**
     * Answer the number of functions used for this call.
     *
     * TUTORIAL: Mandatory method for every call stating the number of usable callback functions.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return core::AbstractGetData3DCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * TUTORIAL: This function should be overloaded if you want to use other callback functions
     * than the default two.
     *
     * @param idx The index of the function to return it's name.
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return core::AbstractGetData3DCall::FunctionName(idx); }

    /** Constructor */
    CallClusteringLoader(void);

    /** Destructor */
    virtual ~CallClusteringLoader(void);


    /**
     * Answer the number of contained spheres
     *
     * @return The size of contained spheres
     */
    SIZE_T Count(void) const;

    HierarchicalClustering::CLUSTERNODE* getLeaves(void) const;


    void SetData(SIZE_T, HierarchicalClustering::CLUSTERNODE*);

    /**
     * Assignment operator
     * Makes a deep copy of all members.
     *
     * TUTORIAL: The assignment operator should always be overloaded for calls.
     * This makes the creation of data modifying modules easier.
     *
     * @param rhs The right hand side operand.
     * @return A referenc to this.
     */
    CallClusteringLoader& operator=(const CallClusteringLoader& rhs);

private:
    /** Number of PNG-Pictures**/
    SIZE_T numberofleaves;

    /**  **/
    HierarchicalClustering::CLUSTERNODE* nodes;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallClusteringLoader> CallClusteringLoaderDescription;

} // namespace MolSurfMapCluster
} // namespace megamol

#endif
