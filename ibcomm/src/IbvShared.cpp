///*
// * IbvShared.cpp
// *
// * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
// * Alle Rechte vorbehalten.
// */
//
//#include "vislib/IbvShared.h"
//
//#include "vislib/AutoLock.h"
//#include "the/system/com_exception.h"
//#include "vislib/StackTrace.h"
//#include "vislib/sysfunctions.h"
//#include "the/trace.h"
//
//
///*
// * vislib::net::ib::IbvShared::IbvShared
// */
//vislib::net::ib::IbvShared::IbvShared(void) {
//    VLSTACKTRACE("IbvShared::IbvShared", __FILE__, __LINE__);
//    HRESULT hr = S_OK;      // Result of API calls.
//    sys::AutoLock lock(IbvShared::lock);
//
//    if (IbvShared::wvProvider == NULL) {
//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Acquiring WinVerbs "
//            "provider...\n");
//        if (FAILED(hr = ::WvGetObject(IID_IWVProvider, 
//                reinterpret_cast<void **>(&IbvShared::wvProvider)))) {
//            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Acquiring WinVerbs "
//                "provider failed with error code %d.\n", hr);
//            throw the::system::com_exception(hr, __FILE__, __LINE__);    
//        }
//
//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Opening completion "
//            "manager...\n");
//        if (FAILED(hr = ::CompManagerOpen(&IbvShared::compMgr))) {
//            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Opening completion "
//                "manager failed with error code %d.\n", hr);
//            sys::SafeRelease(IbvShared::wvProvider);
//            throw the::system::com_exception(hr, __FILE__, __LINE__);
//        }
//
//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Configuring completion "
//            "monitoring...\n");
//        if (FAILED(hr = ::CompManagerMonitor(&IbvShared::compMgr, 
//                IbvShared::wvProvider->GetFileHandle(), 0))) {
//            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Configuring completion "
//                "monitoring failed with error code %d.\n", hr);
//            ::CompManagerClose(&IbvShared::compMgr);
//            sys::SafeRelease(IbvShared::wvProvider);
//            throw the::system::com_exception(hr, __FILE__, __LINE__);
//        }
//
//    } else {
//        IbvShared::wvProvider->AddRef();
//    }
//}
//
//
///*
// * vislib::net::ib::IbvShared::IbvShared
// */
//vislib::net::ib::IbvShared::IbvShared(const IbvShared& rhs) {
//    VLSTACKTRACE("IbvShared::IbvShared", __FILE__, __LINE__);
//    THE_ASSERT(IbvShared::wvProvider != NULL);
//    
//    sys::AutoLock lock(IbvShared::lock);
//    IbvShared::wvProvider->AddRef();
//}
//
//
///*
// * vislib::net::ib::IbvShared::~IbvShared
// */
//vislib::net::ib::IbvShared::~IbvShared(void) {
//    VLSTACKTRACE("IbvShared::~IbvShared", __FILE__, __LINE__);
//    sys::AutoLock lock(IbvShared::lock);
//
//    if (IbvShared::wvProvider != NULL) {
//        if (IbvShared::wvProvider->Release() == 0) {
//            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Releasing shared IB "
//                "resources...\n");
//            ::CompManagerClose(&IbvShared::compMgr);
//            IbvShared::wvProvider = NULL;
//        }
//    }
//}
//
//
///*
// * vislib::net::ib::IbvShared::GetWvProvider
// */
//IWVProvider *vislib::net::ib::IbvShared::GetWvProvider(void) {
//    VLSTACKTRACE("IbvShared::GetWvProvider", __FILE__, __LINE__);
//    sys::AutoLock lock(IbvShared::lock);
//    return IbvShared::wvProvider;
//}
//
//
///*
// * vislib::net::ib::IbvShared::operator =
// */
//vislib::net::ib::IbvShared& vislib::net::ib::IbvShared::operator =(
//        const IbvShared& rhs) {
//    VLSTACKTRACE("IbvShared::operator =", __FILE__, __LINE__);
//    return *this;
//}
//
//
///*
// * vislib::net::ib::IbvShared::compMgr
// */
//COMP_MANAGER vislib::net::ib::IbvShared::compMgr;
//
//
///*
// * vislib::net::ib::IbvShared
// */
//vislib::sys::CriticalSection vislib::net::ib::IbvShared::lock;
//
//
///*
// * vislib::net::ib::IbvShared 
// */
//IWVProvider *vislib::net::ib::IbvShared::wvProvider = NULL;
