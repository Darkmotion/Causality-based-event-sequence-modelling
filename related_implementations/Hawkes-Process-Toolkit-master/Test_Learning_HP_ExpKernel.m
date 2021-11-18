%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comparison of different learning methods of Hawkes processes
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

options.N = 200; % the number of sequences
options.Nmax = 100; % the maximum number of events per sequence
options.Tmax = 100; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 5;
D = 3; % the dimension of Hawkes processes
nTest = 1;
nSeg = 5;
nNum = options.N/nSeg;


disp('Approximate simulation of Hawkes processes via branching process')
disp('Simple exponential kernel')
para1.kernel = 'exp';
para1.w = 1; 
para1.landmark = 0;
L = length(para1.landmark);
para1.mu = rand(D,1)/D;
para1.A = zeros(D, D, L);
for l = 1:L
    para1.A(:,:,l) = (0.5^l)*(0.5+rand(D));
end
para1.A = 0.5*para1.A./max(abs(eig(sum(para1.A,3))));
para1.A = reshape(para1.A, [D, L, D]);
Seqs1 = Simulation_Branch_HP(para1, options);
%Seqs1 = SimulationFast_Thinning_ExpHP(para1, options);


%%
disp('Learning Hawkes processes via different methods')
alg.LowRank = 0;
alg.Sparse = 1;
alg.alphaS = 1;
alg.GroupSparse = 0;
alg.outer = 5;
alg.rho = 0.1;
alg.inner = 8;
alg.thres = 1e-5;
alg.Tmax = [];
alg.storeErr = 0;
alg.storeLL = 0;

Err = zeros(nTest, nSeg);

for n = 1:nTest
    for i = nSeg
       
        
        [A, Phi] = ImpactFunc( para1, options );
        
        disp('Maximum likelihood estimation and basis representation')        
        model1 = Initialization_Basis(Seqs1);
        model1 = Learning_MLE_Basis( Seqs1(1:i*nNum), model1, alg ); 
        [A1, Phi1] = ImpactFunc( model1, options );
        
        
        disp('Least squares and discretization')
        model2.D = D;
        model2.h = 1;
        model2.k = floor(options.M * options.dt/model2.h);
        model2 = Initialization_Discrete(Seqs1);
        model2 = Learning_LS_Discrete( Seqs1(1:i*nNum), model2 );

        
        figure
        title('Simple exponetial kernels')
        for u = 1:D
            for v = 1:D
                subplot(D,D,D*(u-1)+v)
                hold on
                plot(options.dt*(0:(size(Phi,2)-1)), Phi(v,:,u), 'k-')
                plot(options.dt*(0:(size(Phi1,2)-1)), Phi1(v,:,u), 'r-')
                plot(model2.h*(0:(size(model2.A,2)-1)), model2.A(v,:,u), 'b-')
                hold off
                axis tight
                legend('Real', 'MLE', 'LS')%, 'LS2')
                xlabel('Time interval between events')
                ylabel(['\phi', sprintf('%d%d', u, v)])
            end
        end
                
    end
end

