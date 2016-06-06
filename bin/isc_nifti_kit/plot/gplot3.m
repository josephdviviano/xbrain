function [Xout,Yout,Zout]=gplot3(A,xy,lc)
%GPLOT3 Plot graph, as in "graph theory" in three dimensions.
%   GPLOT3(A,xy) plots the graph specified by A and xy. A graph, G, is
%   a set of nodes numbered from 1 to n, and a set of connections, or
%   edges, between them.  
%
%   In order to plot G, two matrices are needed. The adjacency matrix,
%   A, has a(i,j) nonzero if and only if node i is connected to node
%   j.  The coordinates array, xy, is an n-by-3 matrix.
%   
%   GPLOT3(A,xy,LineSpec) uses line type and color specified in the
%   string LineSpec. See PLOT for possibilities.
%
%   [X,Y] = GPLOT3(A,xy) returns the NaN-punctuated vectors
%   X and Y and Z without actually generating a plot. These vectors
%   can be used to generate the plot at a later time if desired.
%   
%   See also SPY, TREEPLOT, GPLOT.

%   By Ali Mohammad Razeghi , 2008.
%   With special thanks to John D'Errico
%   Copyright 2008. 


[i,j,k] = find(A);
[ignore, p] = sort(max(max(i,j),k));
i = i(p);
j = j(p);
k=k(p);

% Create a long, NaN-separated list of line segments,
% rather than individual segments.

X = [ xy(i,1) xy(j,1) repmat(NaN,size(i))]';
Y = [ xy(i,2) xy(j,2) repmat(NaN,size(i))]';
Z = [ xy(i,3) xy(j,3) repmat(NaN,size(i))]';
X = X(:);
Y = Y(:);
Z = Z(:);

if nargout==0,
    if nargin<4,
        plot3(X, Y,Z)
    else
        plot3(X, Y,Z, lc);
    end
else
    Xout = X;
    Yout = Y;
    Zout = Z;
end