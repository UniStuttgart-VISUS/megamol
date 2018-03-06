﻿namespace MegaMolConf.Data.ParamType {

    public sealed class FlexEnum : ParamTypeValueBase {
        public string DefaultValue { get; set; }
        public string[] Values { get; set; }
        public override void ParseDefaultValue(string v) {
            DefaultValue = v;
        }
        public override string DefaultValueString() {
            return DefaultValue;
        }
        public override bool ValuesEqual(string a, string b) {
            return a.Equals(b);
        }
    }

}
