function slBusOut = Header(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    currentlength = length(slBusOut.stamp);
    for iter=1:currentlength
        slBusOut.stamp(iter) = bus_conv_fcns.ros2.msgToBus.builtin_interfaces.Time(msgIn.stamp(iter),slBusOut(1).stamp(iter),varargin{:});
    end
    slBusOut.stamp = bus_conv_fcns.ros2.msgToBus.builtin_interfaces.Time(msgIn.stamp,slBusOut(1).stamp,varargin{:});
    slBusOut.frame_id_SL_Info.ReceivedLength = uint32(strlength(msgIn.frame_id));
    currlen  = min(slBusOut.frame_id_SL_Info.ReceivedLength, length(slBusOut.frame_id));
    slBusOut.frame_id_SL_Info.CurrentLength = uint32(currlen);
    slBusOut.frame_id(1:currlen) = uint8(char(msgIn.frame_id(1:currlen))).';
end
