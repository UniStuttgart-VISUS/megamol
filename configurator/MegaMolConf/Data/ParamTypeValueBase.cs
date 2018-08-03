namespace MegaMolConf.Data {

    public abstract class ParamTypeValueBase : ParamTypeBase {
        abstract public void ParseDefaultValue(string v);
        abstract public string DefaultValueString();
    }

}
