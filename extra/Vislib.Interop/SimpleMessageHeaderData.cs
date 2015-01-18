/*
 * SimpleMessageHeaderData.cs
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;


namespace Vislib.Interop {

    /// <summary>
    /// This is the message header that goes over the wire.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SimpleMessageHeaderData {

        /// <summary>
        /// User-defined message ID.
        /// </summary>
        public UInt32 MessageID;

        /// <summary>
        /// Size of the body to follow in bytes.
        /// </summary>
        public UInt32 BodySize;
    }

}
