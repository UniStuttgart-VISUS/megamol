--
-- vislibcds-dissector.lua
--
-- Copyright (C) 2008 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
-- Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
--

do
    -- VISlib CDS protocol definition
    local protocolVislibCds = Proto("VISlibCDS", "VISlib Cluster Discovery Service Postdissector");
    
    -- Names of magic numbers
    -- Note: Must be network byte order
    local magicNumber = {
		[27766] = "OK (Version 1)"
    }
    local magicNumber2 = {
		[1986225267] = "OK (Version 2)"
    }

    -- Names of builtin messages
    -- Note: Must convert IDs (1, 2, 3, ...) to network byte order in order to work
    local msgIds = {
        [256] = "MSG_TYPE_IAMHERE",
        [512] = "MSG_TYPE_IAMALIVE",
        [768] = "MSG_TYPE_SAYONARA"
    }
    local msgIds2 = {
        [16777216] = "MSG_TYPE_IAMHERE",
        [33554432] = "MSG_TYPE_IAMALIVE",
        [768] = "MSG_TYPE_SAYONARA"
    }    

    -- Fields of a message
    local fieldMagicNumber = ProtoField.uint16("vislibcds.magicnumber", "Magic Number", base.DEC, magicNumber)
    local fieldMagicNumber2 = ProtoField.uint32("vislibcds.magicnumber", "Magic Number Version 2", base.DEC, magicNumber2)
    local fieldMsgId = ProtoField.uint16("vislibcds.msgid", "Message Type", base.DEC, msgIds)
    local fieldMsgId2 = ProtoField.uint32("vislibcds.msgid", "Message Type", base.DEC, msgIds2)
    local fieldResponseAddr = ProtoField.ipv4("vislibcds.responseaddress", "Response Address (Node ID)")
    local fieldResponseAddr6 = ProtoField.ipv6("vislibcds.responseaddress6", "Response Address (Node ID)")
    local fieldResponsePort = ProtoField.uint16("vislibcds.responseport", "Respone Address Port")
    --local fieldSinFamily = ProtoField.int8("vislibcds.sockaddr.family", "Socket Address Family")
    --local fieldSinPort = ProtoField.uint8("vislibcds.sockaddr.port", "Socket Address Port")
    --local fieldSinAddr = ProtoField.ipv4("vislibcds.sockaddr.address", "Socket Address")
    --local fieldSinZero = ProtoField.uint64("vislibcds.sockaddr.padding", "Padding")
    local fieldName = ProtoField.string("vislibcds.name", "Cluster Name")

    protocolVislibCds.fields = {
        fieldMagicNumber,
        fieldMagicNumber2,
        fieldMsgId,
        fieldMsgId2,
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
        subtree:add(fieldMagicNumber2, buffer(0, 4))
        
        local magicNumber = buffer(0, 4):uint()
		if (magicNumber == 1986225267) then
		    -- CDS version 2 (IPv6-compatible)
			subtree:add(fieldMsgId2, buffer(4, 4))
			local msgId = buffer(4, 4):uint()
			local ipVer = buffer(4 + 4, 2):uint()
			
			if (ipVer == 23) then
				subtree:add(fieldResponsePort, buffer(4 + 4 + 2, 2))
				subtree:add(fieldResponseAddr6, buffer(4 + 4 + 8, 16))
			else 
				subtree:add(fieldResponsePort, buffer(4 + 4 + 2, 2))
				subtree:add(fieldResponseAddr, buffer(4 + 4 + 4, 4))
			end
			
			subtree:add(fieldName, buffer(4 + 4 + 128, 256 - 128))
			
		else
			-- CDS version 1 (IPv4 only)
			subtree:add(fieldMsgId, buffer(2, 2))
			local msgId = buffer(2, 2):uint()
        
			if ((msgId == 256) or (msgId == 512) or (msgId == 768)) then
				subtree:add(fieldResponsePort, buffer(4 + 2, 2))
				subtree:add(fieldResponseAddr, buffer(4 + 4, 4))
				subtree:add(fieldName, buffer(4 + 16, 256 - 16))
			end
		end  -- if (magicNumber == 1986225267)
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

    end  -- function protocolVislibCds.dissector

    -- Register the protocol on VISlib CDS default port 28181
    tableUdpPorts = DissectorTable.get("udp.port")
    tableUdpPorts:add(28181, protocolVislibCds)
end
