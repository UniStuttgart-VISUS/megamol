using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data.ParamType {

    public sealed class Bool : ParamTypeValueBase {
        public bool DefaultValue { get; set; }
        public bool ParseValue(string v) {
            if (v.Equals("true", StringComparison.InvariantCultureIgnoreCase)) return true;
            else if (v.Equals("t", StringComparison.InvariantCultureIgnoreCase)) return true;
            else if (v.Equals("yes", StringComparison.InvariantCultureIgnoreCase)) return true;
            else if (v.Equals("y", StringComparison.InvariantCultureIgnoreCase)) return true;
            else if (v.Equals("on", StringComparison.InvariantCultureIgnoreCase)) return true;
            else if (v.Equals("false", StringComparison.InvariantCultureIgnoreCase)) return false;
            else if (v.Equals("f", StringComparison.InvariantCultureIgnoreCase)) return false;
            else if (v.Equals("no", StringComparison.InvariantCultureIgnoreCase)) return false;
            else if (v.Equals("n", StringComparison.InvariantCultureIgnoreCase)) return false;
            else if (v.Equals("off", StringComparison.InvariantCultureIgnoreCase)) return false;
            else return (int.Parse(v) != 0);
        }
        public override void ParseDefaultValue(string v) {
            DefaultValue = ParseValue(v);
        }
        public override string DefaultValueString() {
            return this.DefaultValue.ToString();
        }
        public override bool ValuesEqual(string a, string b) {
            return ParseValue(a).Equals(ParseValue(b));
        }
    }

}
