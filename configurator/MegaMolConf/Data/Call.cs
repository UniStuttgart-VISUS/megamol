using System.Diagnostics;

namespace MegaMolConf.Data {
    [DebuggerDisplay("Call {Name}")]
    public sealed class Call {
        public string Name { get; set; }
        public string Description { get; set; }
        public string[] FunctionName { get; set; }
    }
}
