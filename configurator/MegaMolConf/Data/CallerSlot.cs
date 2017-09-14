using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("CallerSlot {Name}")]
    public sealed class CallerSlot {
        public string Name { get; set; }
        public string Description { get; set; }
        public string[] CompatibleCalls { get; set; }
    }
}
