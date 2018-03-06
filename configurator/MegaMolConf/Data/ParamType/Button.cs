namespace MegaMolConf.Data.ParamType {

    public sealed class Button : ParamTypeBase {
        public override bool ValuesEqual(string a, string b) {
            return true;
        }
    }

}
