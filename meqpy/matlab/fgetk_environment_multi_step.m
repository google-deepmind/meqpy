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
function [LY,Stop,State,solverinfo_sub,LY_sub] = fgetk_environment_multi_step(State,Actions,L,LX,num_steps)
% Multi-step helper function. Steps fgetk_environment with fixed actions and checks for terminations at each step.
% Returns the LY of the last iteration.
% Actions: actions from the controller/agent
% Parameters: Struct containing L and LX structures.
% num_steps: Number of simulation steps to take. Can be 1.

% this LY is the last one of the previous call,
% because fgetk_environment returns the previous LY
[LY,Stop,State,solverinfo] = fgetk_environment(State,Actions,L,meqxk(LX,1),[],'step');

% initialize memory
LY_sub = repmat({LY},1,num_steps);
solverinfo_sub = repmat({solverinfo},1,num_steps);

for it=2:num_steps
  [LY_sub{it},Stop,State,solverinfo] = fgetk_environment(State, Actions,L,meqxk(LX,it),[],'step');
  solverinfo_sub{it} = solverinfo;
  if Stop, return; end
end

end