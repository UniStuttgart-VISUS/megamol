--
-- vislibcds-dissector.lua
--
-- Copyright (C) 2008 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
-- Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
--

do
    -- VISlib CDS protocol definition
    local protocolVislibCds = Proto("VISlibCDS", "VISlib Cluster Discovery Service Postdissector");

    -- Names of builtin messages
    -- Note: Must convert IDs (1, 2, 3, ...) to network byte order in order to work
    local msgIds = {
        [256] = "MSG_TYPE_IAMHERE",
        [512] = "MSG_TYPE_IAMALIVE",
        [768] = "MSG_TYPE_SAYONARA"
    }

    -- Fields of a message
    local fieldMagicNumber = ProtoField.uint16("vislibcds.magicnumber", "Magic Number", base.DEC)
    local fieldMsgId = ProtoField.uint16("vislibcds.msgid", "Message Type", base.DEC, msgIds)
    local fieldResponseAddr = ProtoField.ipv4("vislibcds.responseaddress", "Response Address (Node ID)")
    local fieldResponsePort = ProtoField.uint16("vislibcds.responseport", "Respone Address Port")
    --local fieldSinFamily = ProtoField.int8("vislibcds.sockaddr.family", "Socket Address Family")
    --local fieldSinPort = ProtoField.uint8("vislibcds.sockaddr.port", "Socket Address Port")
    --local fieldSinAddr = ProtoField.ipv4("vislibcds.sockaddr.address", "Socket Address")
    --local fieldSinZero = ProtoField.uint64("vislibcds.sockaddr.padding", "Padding")
    local fieldName = ProtoField.string("vislibcds.name", "Cluster Name")

    protocolVislibCds.fields = {
        fieldMagicNumber,
        fieldMsgId,
        fieldResponseAddr,
        fieldResponsePort,
        fieldName
    }

    --local fieldBody = ProtoField.string("vislibcds.body", "Body")
--        local data_dis = Dissector.get("data")

    --
    -- This method adds the dissector to the protocl tree.
    --
    function protocolVislibCds.dissector(buffer, pinfo, tree)
        pinfo.cols.protocol = "VISlibCDS"
        local subtree = tree:add(protocolVislibCds, buffer(), "VISlib Cluster Discovery Service Packet")
        subtree:add(fieldMagicNumber, buffer(0, 2))
        subtree:add(fieldMsgId, buffer(2, 2))

        local msgId = buffer(2, 2):uint()
        if ((msgId == 256) or (msgId == 512) or (msgId == 768)) then
            subtree:add(fieldResponsePort, buffer(4 + 2, 2))
            subtree:add(fieldResponseAddr, buffer(4 + 4, 4))
            subtree:add(fieldName, buffer(4 + 16, 256 - 16))
        end
--        local t = root:add(protocolVislibCds, buf(0, 2))
--        t:add(fieldMagicNumber, buf(0, 2))
--        t:add(fieldMsgId, buf(0, 2))
--                t:add(f_dir,buf(1,1))

--                local proto_id = buf(0,1):uint()

--                local dissector = protos[proto_id]

--                if dissector ~= nil then
--                        dissector:call(buf(2):tvb(),pkt,root)
--                elseif proto_id < 2 then
--                        t:add(f_text,buf(2))
--                        -- pkt.cols.info:set(buf(2,buf:len() - 3):string())
--                else
--                        data_dis:call(buf(2):tvb(),pkt,root)
--                end 

    end  -- function trivial_proto.dissector

    -- Register the protocol on VISlib CDS default port 28181
    tableUdpPorts = DissectorTable.get("udp.port")
    tableUdpPorts:add(28181, protocolVislibCds)

--local wtap_encap_table = DissectorTable.get("wtap_encap")
--local udp_encap_table = DissectorTable.get("udp.port")
--wtap_encap_table:add(wtap.USER15,p_multi)
--wtap_encap_table:add(wtap.USER12,p_multi)
--udp_encap_table:add(7555,p_multi)
end
