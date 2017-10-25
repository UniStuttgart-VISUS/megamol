using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ZeroMQ;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Models the connection to an instance of the MegaMol Lua Remote
    /// </summary>
    /// <remarks>Implemented using a ZeroMQ Request Socket</remarks>
    /// <remarks>originally from the SimpleParamRemote by S. Grottel</remarks>
    public class Connection : IDisposable {

        /// <summary>
        /// The default timeout
        /// </summary>
        private static TimeSpan defaultTimeout = TimeSpan.FromSeconds(1.0);

        /// <summary>
        /// Gets the default timeout
        /// </summary>
        public static TimeSpan DefaultTimeout {
            get { return defaultTimeout; }
        }

        /// <summary>
        /// Creates a connection.
        /// </summary>
        /// <param name="adress">The adress to connect to. Use ZeroMQ connection string syntax, e.g. "tcp://localhost:35421"</param>
        /// <returns>The new connection</returns>
        /// <remarks>The connection might be established asynchronous.
        /// Errors might occur at the actual communication.</remarks>
        public static Connection Connect(string adress) {
            return Connect(adress, DefaultTimeout);
        }

        /// <summary>
        /// Creates a connection.
        /// </summary>
        /// <param name="adress">The adress to connect to. Use ZeroMQ connection string syntax, e.g. "tcp://localhost:35421"</param>
        /// <param name="timeout">The timeout when the function will throw an exception</param>
        /// <returns>The new connection</returns>
        /// <remarks>The connection might be established asynchronous.
        /// Errors might occur at the actual communication.</remarks>
        public static Connection Connect(string adress, TimeSpan timeout) {
            ZContext c = ZeroMQContext.Get;
            ZSocket s = new ZSocket(c, ZSocketType.REQ);
            ZError e;
            if (!s.Connect(adress, out e)) {
                throw new Exception(e.ToString());
            }

            s.ReceiveTimeout = timeout;
            s.SendTimeout = timeout;

            return new Connection() { context = c, socket = s };
        }

        private Connection() { }

        #pragma warning disable 414
        private ZContext context; // to keep alive
        #pragma warning restore 414
        private ZSocket socket;

        /// <summary>
        /// Sends a requestion and waits for the response
        /// </summary>
        /// <param name="req">The request to be sent to MegaMol</param>
        /// <returns>The response answer by MegaMol</returns>
        public Response Send(Request req) {
            return Send(req, defaultTimeout);
        }

        /// <summary>
        /// Sends a requestion and waits for the response
        /// </summary>
        /// <param name="req">The request to be sent to MegaMol</param>
        /// <param name="timeout">The timeout when the function will throw an exception</param>
        /// <returns>The response answer by MegaMol</returns>
        public Response Send(Request req, TimeSpan timeout) {
            if (req == null) throw new ArgumentNullException("req");

            socket.ReceiveTimeout = timeout;
            socket.SendTimeout = timeout;

            ZFrame reqData = req.MakeZMQRequest();
            if (reqData == null) throw new ArgumentException("req seemed illegal");
            socket.Send(reqData);

            Response resp = null;
            using (ZFrame reply = socket.ReceiveFrame()) {
                Response r = new Response();
                r.Request = req; // set request first, because that object is required to parse the answer
                r.fromZFrameString(reply.ReadString(Encoding.UTF8));
                resp = r;
            }

            return resp;
        }

        /// <summary>
        /// Answer if the socket of this connection is valid. This is no guarantee that the connection is establed.
        /// </summary>
        public bool Valid {
            get { return socket != null; }
        }

        /// <summary>
        /// Closes this connection. This connection object can then no longer be used
        /// </summary>
        public void Close() {
            ZSocket s = socket;
            socket = null;
            if (s != null) {
                s.Close();
                s.Dispose();
                s = null;
            }
            // we keep the context, because another connection might be opened right away
        }

        /// <summary>
        /// Disposes this connection
        /// </summary>
        public void Dispose() {
            ZSocket s = socket;
            socket = null;
            if (s != null) {
                s.Close();
                s.Dispose();
                s = null;
            }
            context = null;
        }

    }

}
