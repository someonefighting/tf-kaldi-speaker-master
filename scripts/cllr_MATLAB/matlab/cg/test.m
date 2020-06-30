function y = test(x);
q = 0.1;
y = -q*log(sigmoid(x))-(1-q)*log(sigmoid(-x));