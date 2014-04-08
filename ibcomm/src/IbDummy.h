/*
 * IbDummy.h
 *
 * Copyright (C) 2006 - 2014 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBDUMMY_H_INCLUDED
#define VISLIB_IBDUMMY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace vislib {
namespace ibcomm {


    /**
     * This class forces the library to be built in case the OFED SDK is not
     * available on the building machine.
     */
    class IbDummy {

    public:

        /** Ctor. */
        IbDummy(void);

        /** Dtor. */
        ~IbDummy(void);

    protected:

    private:

    };
    
} /* end namespace ibcomm */
} /* end namespace vislib */

#endif /* VISLIB_IBDUMMY_H_INCLUDED */

