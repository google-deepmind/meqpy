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
function [LY,LY_substeps] = ...
  fgetk_environment_loop(Actions,L,LX,dt,num_steps)
% Looping function fgetk_environment_loop
% Runs fgetk_environment in open-loop
%
%  LY = fgetk_environment_loop(L, LX, Va); % multistep
% [LY,LY_substeps] = ...
%         fgetk_environment_loop(L, LX, Va,2); % multistep
%
% Inputs:
% Parameters: environment parameters containing L,LX,action_labels
% Actions: Time sequence of Actions (open-loop run)
% num_steps (optional) number of time steps per call (for mult-stepping). Default=1;

dodebug = L.P.debug;
nt = numel(LX.t); % max times
if size(Actions,2) < nt
  fprintf('Number of actions supplied is shorter than timesteps in LX')
end

if nargin < 5
  num_steps = 1; % no multistepping by default
end
assert(num_steps>0,'num_steps must be positive')

multi_step = (num_steps>1); % flag for multi-step call

%% initialize

% init fgetk environment
[LY0,Stop,State,solverinfo] = fgetk_environment([],[],L,meqxk(LX,1),dt,'init');

% create buffer for LY
num_macro_steps = ceil((nt-1)/num_steps);
LY = repmat(LY0,num_macro_steps,1);

if multi_step && nargout>1
  % init
  LY_substeps = repmat({LY0},num_steps,num_macro_steps);
  solverinfo_sub = repmat({solverinfo},num_steps,num_macro_steps);
end

%% Run time loop
if ~multi_step
  for it=2:nt
    if dodebug; fprintf('.'); end
    this_Action = Actions(:, it);
    % single-step call
    % it-1 because the environment returns the state at the previous time step
    [LYp,Stop,State] = fgetk_environment(State,this_Action,L,meqxk(LX,it));
    LY(it-1) = LYp;

    if Stop
      fprintf('Stopping due to internal stop criterion:\n %\n', Stop)
      break
    end
  end
else
  for it=2:num_steps:nt-1
    % multi-step call
    this_Action = Actions(:, it);
    jj = ceil(it/num_steps);

    LXs = meqxk(LX,it+(0:num_steps-1)); % LX slices for these steps
    [LY(jj), Stop, State, solverinfo_sub(:,jj), LY_substeps(:,jj)] = ...
      fgetk_environment_multi_step(State,this_Action,L,LXs,num_steps);
     if Stop
      fprintf('Stopping due to internal stop criterion:\n %\n', Stop)
      break
    end
  end
  LY_substeps_array = cell2mat(LY_substeps);
  LY_substeps = orderfields(meqlpack(LY_substeps_array(:)));
end

LY = orderfields(meqlpack(LY));

end