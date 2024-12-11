function slBusOut = MultiArrayDimension(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    slBusOut.label_SL_Info.ReceivedLength = uint32(strlength(msgIn.label));
    currlen  = min(slBusOut.label_SL_Info.ReceivedLength, length(slBusOut.label));
    slBusOut.label_SL_Info.CurrentLength = uint32(currlen);
    slBusOut.label(1:currlen) = uint8(char(msgIn.label(1:currlen))).';
    slBusOut.size = uint32(msgIn.size);
    slBusOut.stride = uint32(msgIn.stride);
end
