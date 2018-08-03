using System;
using ZeroMQ;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Singleton manager for ZeroMQ context
    /// </summary>
    /// <remarks>originally from the SimpleParamRemote by S. Grottel</remarks>
    internal class ZeroMQContext {

        private static WeakReference<ZContext> singleton = new WeakReference<ZContext>(null);
        private static object threadlock = new object();

        private ZeroMQContext() { }

        public static ZContext Get {
            get {
                lock (threadlock) {
                    ZContext c = null;
                    if (!singleton.TryGetTarget(out c)) {
                        c = new ZContext();
                        singleton = new WeakReference<ZContext>(c);
                    }
                    return c;
                }
            }
        }

    }

}
