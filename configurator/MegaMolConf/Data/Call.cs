using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("Call {Name}")]
    public sealed class Call {
        public string Name { get; set; }
        public string Description { get; set; }
        public string[] FunctionName { get; set; }
    }
}
