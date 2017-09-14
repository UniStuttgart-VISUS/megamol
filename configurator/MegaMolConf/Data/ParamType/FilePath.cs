using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data.ParamType {

    public sealed class FilePath : ParamTypeValueBase {
        public string DefaultValue { get; set; }
        public override void ParseDefaultValue(string v) {
            this.DefaultValue = v;
        }
        public override string DefaultValueString() {
            return this.DefaultValue;
        }
        public override bool ValuesEqual(string a, string b) {
            return a.Equals(b);
        }
    }

}
