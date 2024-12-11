function slBusOut = JointState(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    currentlength = length(slBusOut.header);
    for iter=1:currentlength
        slBusOut.header(iter) = bus_conv_fcns.ros2.msgToBus.std_msgs.Header(msgIn.header(iter),slBusOut(1).header(iter),varargin{:});
    end
    slBusOut.header = bus_conv_fcns.ros2.msgToBus.std_msgs.Header(msgIn.header,slBusOut(1).header,varargin{:});
    maxlength = length(slBusOut.name);
    recvdlength = length(msgIn.name);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'name', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.name_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.name_SL_Info.CurrentLength = uint32(currentlength);
    for iter=1:currentlength
        recvlen = strlength(msgIn.name(iter));
        maxlen = length(slBusOut.name(iter).data);
        curlen = min(recvlen, maxlen);
        if (max(recvlen) > maxlen) && ...
                isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
            diag = MSLDiagnostic([], ...
                                 message('ros:slros:busconvert:TruncatedArray', ...
                                         'name', msgIn.MessageType, maxlen, max(recvdlength), maxlength, varargin{2}));
            reportAsWarning(diag);
        end
        slBusOut.name(iter).data_SL_Info.CurrentLength = uint32(curlen);
        slBusOut.name(iter).data_SL_Info.ReceivedLength = uint32(recvlen);
        slBusOut.name(iter).data(1:curlen) = uint8(char(msgIn.name(iter)));
    end
    maxlength = length(slBusOut.position);
    recvdlength = length(msgIn.position);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'position', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.position_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.position_SL_Info.CurrentLength = uint32(currentlength);
    slBusOut.position = double(msgIn.position(1:slBusOut.position_SL_Info.CurrentLength));
    if recvdlength < maxlength
    slBusOut.position(recvdlength+1:maxlength) = 0;
    end
    maxlength = length(slBusOut.velocity);
    recvdlength = length(msgIn.velocity);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'velocity', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.velocity_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.velocity_SL_Info.CurrentLength = uint32(currentlength);
    slBusOut.velocity = double(msgIn.velocity(1:slBusOut.velocity_SL_Info.CurrentLength));
    if recvdlength < maxlength
    slBusOut.velocity(recvdlength+1:maxlength) = 0;
    end
    maxlength = length(slBusOut.effort);
    recvdlength = length(msgIn.effort);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'effort', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.effort_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.effort_SL_Info.CurrentLength = uint32(currentlength);
    slBusOut.effort = double(msgIn.effort(1:slBusOut.effort_SL_Info.CurrentLength));
    if recvdlength < maxlength
    slBusOut.effort(recvdlength+1:maxlength) = 0;
    end
end
