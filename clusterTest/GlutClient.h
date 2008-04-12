/*
 * GlutClient.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLUTCLIENT_H_INCLUDED
#define VISLIBTEST_GLUTCLIENT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/GlutClientNode.h"


class GlutClient : public vislib::net::cluster::GlutClientNode<GlutClient> {

public:

    virtual ~GlutClient(void);

protected:

    GlutClient(void);

    friend class vislib::net::cluster::GlutClusterNode<GlutClient>;

};

#endif /* VISLIBTEST_GLUTCLIENT_H_INCLUDED */
