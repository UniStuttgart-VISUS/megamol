using System.Text;
using ZeroMQ;

namespace MegaMolConf.Communication {

    /// <summary>
    /// A request to be send to MegaMol LR Host
    /// </summary>
    public abstract class Request {

        /// <summary>
        /// The command string to be sent
        /// </summary>
        public abstract string Command { get; }

        internal ZFrame MakeZMQRequest() {
            return new ZFrame(Command, Encoding.UTF8);
        }

        internal virtual object parseAnswerFromZString(string v) {
            return v;
        }

    }

}
