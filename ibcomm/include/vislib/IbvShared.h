///*
// * IbvShared.h
// *
// * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
// * Alle Rechte vorbehalten.
// */
//
//#ifndef VISLIB_IBVSHARED_H_INCLUDED
//#define VISLIB_IBVSHARED_H_INCLUDED
//#if (defined(_MSC_VER) && (_MSC_VER > 1000))
//#pragma once
//#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
//#if defined(_WIN32) && defined(_MANAGED)
//#pragma managed(push, off)
//#endif /* defined(_WIN32) && defined(_MANAGED) */
//
//
//#include "vislib/Socket.h"              // Must be first.
//#include "vislib/CriticalSection.h"
//#include "vislib/SmartRef.h"
//
//#include "rdma/winverbs.h"
//
//#include "comp_channel.h"
//
//
//namespace vislib {
//namespace net {
//namespace ib {
//
//
//    /**
//     * Manages resources shared by different instances of IB classes, like the
//     * IWvProvider.
//     *
//     * This class is thread-safe.
//     */
//    class IbvShared {
//
//    public:
//
//        /** Ctor. */
//        IbvShared(void);
//
//        IbvShared(const IbvShared& rhs);
//
//        /** Dtor. */
//        virtual ~IbvShared(void);
//
//        IWVProvider *GetWvProvider(void);
//
//        IbvShared& operator =(const IbvShared& rhs);
//
//    private:
//
//        /** Completion manager. */
//        static COMP_MANAGER compMgr;
//
//        /** The lock protecting the static resources. */
//        static sys::CriticalSection lock;
//
//        /** The only instance of the object. */
//        static IWVProvider *wvProvider;
//    };
//    
//} /* end namespace ib */
//} /* end namespace net */
//} /* end namespace vislib */
//
//#if defined(_WIN32) && defined(_MANAGED)
//#pragma managed(pop)
//#endif /* defined(_WIN32) && defined(_MANAGED) */
//#endif /* VISLIB_IBVSHARED_H_INCLUDED */
