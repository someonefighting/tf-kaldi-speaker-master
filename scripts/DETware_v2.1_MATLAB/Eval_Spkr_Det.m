%------------------------------
%load speaker detection output scores
% load true_speaker_scores
% load impostor_scores
best;
true_speaker_scores=target
impostor_scores=nontarget
%------------------------------
%initialize the DCF parameters
Set_DCF (10, 1, 0.01);

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa] = Compute_DET (true_speaker_scores, impostor_scores);

%------------------------------
%plot results

% Set tic marks
Pmiss_min = 0.01;
Pmiss_max = 0.7;
Pfa_min = 0.01;
Pfa_max = 0.7;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

%call figure, plot DET-curve
figure;
Plot_DET (P_miss, P_fa,'r');
title ('Speaker Detection Performance');

hold on;

%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
Set_DCF(C_miss,C_fa,P_target);
[DCF_opt Popt_miss Popt_fa] = Min_DCF(P_miss,P_fa);
% Plot_DET (Popt_miss,Popt_fa,'ko');

%% load impostor_scores
student;
true_speaker_scores=target
impostor_scores=nontarget
%------------------------------
%initialize the DCF parameters
Set_DCF (10, 1, 0.01);

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa] = Compute_DET (true_speaker_scores, impostor_scores);

%------------------------------
%plot results

% Set tic marks
Pmiss_min = 0.01;
Pmiss_max = 0.45;
Pfa_min = 0.01;
Pfa_max = 0.45;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

%call figure, plot DET-curve
hold on;
Plot_DET (P_miss, P_fa,'b');

hold on;

%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
Set_DCF(C_miss,C_fa,P_target);
[DCF_opt Popt_miss Popt_fa] = Min_DCF(P_miss,P_fa);
% Plot_DET (Popt_miss,Popt_fa,'ko');
%%------------------------------
%load speaker detection output scores
% load true_speaker_scores
% %% load impostor_scores
teacher;
true_speaker_scores=target
impostor_scores=nontarget
%------------------------------
%initialize the DCF parameters
Set_DCF (10, 1, 0.01);

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa] = Compute_DET (true_speaker_scores, impostor_scores);

%------------------------------
%plot results

% Set tic marks
Pmiss_min = 0.01;
Pmiss_max = 0.45;
Pfa_min = 0.01;
Pfa_max = 0.45;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

%call figure, plot DET-curve
hold on;
Plot_DET (P_miss, P_fa,'c');

hold on;

%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
Set_DCF(C_miss,C_fa,P_target);
[DCF_opt Popt_miss Popt_fa] = Min_DCF(P_miss,P_fa);
% Plot_DET (Popt_miss,Popt_fa,'ko');

%% load impostor_scores
xvector;
true_speaker_scores=target
impostor_scores=nontarget
%------------------------------
%initialize the DCF parameters
Set_DCF (10, 1, 0.01);

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa] = Compute_DET (true_speaker_scores, impostor_scores);

%------------------------------
%plot results

% Set tic marks
Pmiss_min = 0.01;
Pmiss_max = 0.45;
Pfa_min = 0.01;
Pfa_max = 0.45;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

%call figure, plot DET-curve
hold on;
Plot_DET (P_miss, P_fa,'g');

hold on;

%find lowest cost point and plot
C_miss = 1;
C_fa = 1;
P_target = 0.5;
Set_DCF(C_miss,C_fa,P_target);
[DCF_opt Popt_miss Popt_fa] = Min_DCF(P_miss,P_fa);
% Plot_DET (Popt_miss,Popt_fa,'ko');
%  legend("xvector","","teacher","","student","","xvector+ryhthm variation","")
 legend("Student-64","Student-32","X-vector+A-softmax+Rhythm features","X-vector baseline","")
