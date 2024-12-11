function slBusOut = Float64MultiArray(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    currentlength = length(slBusOut.layout);
    for iter=1:currentlength
        slBusOut.layout(iter) = bus_conv_fcns.ros2.msgToBus.std_msgs.MultiArrayLayout(msgIn.layout(iter),slBusOut(1).layout(iter),varargin{:});
    end
    slBusOut.layout = bus_conv_fcns.ros2.msgToBus.std_msgs.MultiArrayLayout(msgIn.layout,slBusOut(1).layout,varargin{:});
    maxlength = length(slBusOut.data);
    recvdlength = length(msgIn.data);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'data', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.data_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.data_SL_Info.CurrentLength = uint32(currentlength);
    slBusOut.data = double(msgIn.data(1:slBusOut.data_SL_Info.CurrentLength));
    if recvdlength < maxlength
    slBusOut.data(recvdlength+1:maxlength) = 0;
    end
end
