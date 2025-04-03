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
classdef fgetk_environment_test < meq_test
  % Integration test for fgetk_environment. Generate Parameters with L,LX. Step environment in open loop


  properties
    verbose = 1;
  end
  properties(TestParameter)
      tok    = struct('ana2',      'ana','singlet1',   'tcv'   ,'singlet2','tcv'       ,'singlet3','tcv'       ,'droplet1','tcv'      ,'doublet1',          'tcv');
      shot   = struct('ana2',          2,'singlet1',   66028   ,'singlet2',70166       ,'singlet3',65668       ,'droplet1',67151      ,'doublet1',          81152);
      t0     = struct('ana2',          0,'singlet1',    0.45   ,'singlet2',0.0872      ,'singlet3',0.2         ,'droplet1',0.03       ,'doublet1',           0.01);
      cde    = struct('ana2','cde_ss_0D','singlet1','cde_ss_0D','singlet2','cde_ss_0D' ,'singlet3','cde_ss_0D' ,'droplet1','cde_ss_0D','doublet1','cde_OhmTor_1D');
    idoublet = struct('ana2',      false,'singlet1',   false   ,'singlet2', false      ,'singlet3',false       ,'droplet1',true       ,'doublet1',           true);
    eqsource = struct('ana2',      'fbt','singlet1',   'liu'   ,'singlet2', 'liu'      ,'singlet3','liu'       ,'droplet1','fbt'      ,'doublet1',          'liu');
  end

  methods(Test,TestTags={'TCV'},ParameterCombination = 'sequential')

    function test_fgetk_environment(testCase,tok,shot,t0,eqsource,cde,idoublet)
      [ok,msg] = meq_test.check_tok(tok,shot);
      testCase.assumeTrue(ok,msg);

      dt = 1e-5; nt = 6;
      time = t0 + (0:dt:nt*dt);

      PP_user = {'mkryl',100,'tolF',1e-4,'anajac',true,...
        'debug',testCase.verbose,'iterq',20}; % custom FGE parameters for this test
      switch tok
        case 'ana'
          % just build parameters directly
          [L,LX] = fge(tok,shot,t0,PP_user{:});
        case 'tcv'
          % use fgetk_get_parameter function
          % Test function for fgetk_environment - shows an example open-loop run
          doplot = testCase.verbose>1;
          t_evo = 0; % don't evolve FGE first
          testCase.applyFixture(meqscripts_fixture())% need scripts folder
          [~,~,~,L,LX] = fge_init_from_liu_or_fbt(shot,time(1),eqsource,cde,idoublet,t_evo,doplot,{},PP_user); % return directly, no storing
      end

      % copy fge LX to multiple time slices
      LX = fgex(tok,time,L,LX);

      % default call - baseline for test
      PF_voltage = -1000;
      LX.Va(:) = PF_voltage; % set some interesting voltages for t his test
      LYo = fget(L,LX); LYo = meqxk(LYo,1:numel(LYo.t)-1); % chuck last one for comparison

      %% Test iterative call using fgetk environment loop
      n_actions = L.G.na;
      Actions = PF_voltage*ones(n_actions, numel(LX.t)); % Voltages are actions as seen by agent
      LY = fgetk_environment_loop(Actions,L,LX,dt);

      % Checks
      testCase.verifyTrue(meq_test.check_convergence(L,[],LY), 'Not all time slices converged');
      testCase.verifyTrue(all(sum(isfinite(LY.q95),1)==LY.nB),'number of valid q95 values must match number of domains')
      % Check equality w.r.t. standard FGET call
      testCase.verifyEqual(LY.Fx,LYo.Fx,'AbsTol',L.Fx0*sqrt(eps));
      testCase.verifyEqual(LY.aq,LYo.aq,'AbsTol',L.P.r0*sqrt(eps));
      testCase.verifyEqual(LY.iqQ,LYo.iqQ,'AbsTol',L.P.r0*sqrt(eps));
      testCase.verifyEqual(LY.Iu,LYo.Iu,'AbsTol',L.Iu0*sqrt(eps));

      %% multi loop test
      if shot == testCase.shot.ana2 % test multiloop only on anamak
        num_steps = 2;
        [LY,LY_all] = fgetk_environment_loop(Actions,L,LX,dt,num_steps);
        % LY_all should contain all slices
        ignoredFields = {'FW','tokamak','shot'};
        testCase.verifyTrue(structcmp(LYo,meqxk(LY_all,1:numel(LY_all.t)),sqrt(eps),ignoredFields))

        % LY should contain subset of LY_all
        testCase.verifyTrue(...
          structcmp(LY,meqxk(LY_all,1:num_steps:numel(LY_all.t)),sqrt(eps),...
          ignoredFields),... % don't check these
          'LY does not contain expected subset');
      end
    end
  end
end

