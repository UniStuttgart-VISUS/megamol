using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("CalleeSlot {Name}")]
    public sealed class CalleeSlot {
        public string Name { get; set; }
        public string Description { get; set; }
        public string[] CompatibleCalls { get; set; }
    }
}
