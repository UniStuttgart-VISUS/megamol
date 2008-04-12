/*
 * AbstractControlledNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED
#define VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClusterNode.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * TODO: comment class
     */
    class AbstractControlledNode : public virtual AbstractClusterNode {

    public:

        /** Dtor. */
        virtual ~AbstractControlledNode(void);

    protected:

        /** Ctor. */
        AbstractControlledNode(void);

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED */

