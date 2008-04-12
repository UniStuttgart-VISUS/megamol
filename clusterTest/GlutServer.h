/*
 * GlutServer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLUTSERVER_H_INCLUDED
#define VISLIBTEST_GLUTSERVER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/GlutServerNode.h"


class GlutServer : public vislib::net::cluster::GlutServerNode<GlutServer> {

public:

    virtual ~GlutServer(void);

protected:

    GlutServer(void);

    friend class vislib::net::cluster::GlutClusterNode<GlutServer>;

};

#endif /* VISLIBTEST_GLUTSERVER_H_INCLUDED */
