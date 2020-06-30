function [val,state] = direval(x,state);

x0 = state.direval.x0;
dir = state.direval.dir;
f = state.direval.f;

[val,state] = feval(f,x0+x*dir);