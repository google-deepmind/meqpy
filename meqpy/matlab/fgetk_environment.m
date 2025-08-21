% Copyright 2024 The meqpy Authors.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
function [LY, Stop, State, solverinfo] = fgetk_environment(State,Actions,L,LXt,dt,callmethod)
% 'Environment' implementation of fgetk e.g. for learning purposes
% [LY, Stop, State,solverinfo] = fgetk_environment(State,Actions,L,LXt,dt,callmethod)
%
% Actions: voltage setpoints from controller/agent (in L.G.dima order)
% L: FGE L structure
% LX: single slice of FGE LX structure
% dt: time step [s]
% callmethod: 'init' for initialization, 'step or leave empty for stepping
%
% Outputs: LY: MEQ equilibrium output structure
% Stop: either an empty string or a termination reason
% State: structure with internal fgetk_environment state
% solverinfo: Solver info structure from solveF.m
%
% NB: environment returns one-step-delayed LY to mimick the situation in
% the real plant


if nargin == 4; callmethod = 'step'; end

switch callmethod
  case 'init'
    assert(isempty(State),'call with empty State when calling init callmethod')
    assert(numel(LXt.t)==1,'must initialize with a single LX slice')
    % Initialize fget states
    [LY,LYD,xnl,xnldot,Prec,psstate,dstate,Tstate,nnoc,solverinfo] = ...
      fget(L,LXt,'dt',dt);

    % Store initial state structure
    State.it = 1;
    State.Prec = Prec;
    State.xnl = xnl;
    State.xnldot = xnldot;
    State.Tstate = Tstate;
    State.PSstate = psstate;
    State.dstate = dstate;
    State.nnoc = nnoc;
    State.LYt = LY;
    State.dt  = dt;
    alarm = false;

  case 'step'
    if isempty(State), error('State can not be empty for step call'); end
    % Step
    % Get non-controlled inputs for this iteration counter from LX
    State.it = State.it + 1; % increment iteration counter

    assert(abs(LXt.t - State.LYt.t - State.dt) < 1e-12, ...
      'dt in LX (%e) - (%e) does not match initial dt (%e). dt needs to be constant.', ...
          LXt.t, State.LYt.t, State.dt)

    % Assign voltages from Actions
    LXt.Va = Actions;

    LY = State.LYt; % Must also return previous LY
    % FGE time step call
    [State.LYt,~,State.xnl,State.xnldot,State.Prec,...
      State.PSstate,State.dstate,State.Tstate,State.nnoc,alarm,solverinfo] = ... % assigning new states
      fget(L,LXt,...
      'LYp',State.LYt,... % previous LY
      'xnl',State.xnl,...
      'xnldot',State.xnldot,...
      'Prec',State.Prec,...
      'psstate',State.PSstate,...
      'dstate',State.dstate,...
      'Tstate',State.Tstate,...
      'nnoc',State.nnoc,...
      'dt',State.dt...
      );

  otherwise
    error('unexpected call method %s',callmethod)
end

State.Vact = LY.Va;
Stop = ''; % init
% Consider a high residual as no solution.
if State.LYt.res > 1e2
  Stop = sprintf('No solution found, residual > 1e2. Residual: %3.3f', State.LYt.res);
elseif alarm
  Stop = sprintf('Power supply alarm, coil current limits exceeded.');
elseif State.nnoc > L.P.nnoc
  Stop = sprintf('Simulator did not converge for %d subsequent steps, maximum(P.nnoc)=%d.', State.nnoc,L.P.nnoc);
end

end